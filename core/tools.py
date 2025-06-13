from __future__ import annotations
from dataclasses import dataclass
import traceback
from typing import Callable, Awaitable, Type
from pydantic import BaseModel, Field, create_model
import inspect
import asyncio
from functools import partial, wraps
from typing import Optional, List, Dict, Any, Tuple
import os
import sqlite3
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape


SCHEMA_PATH = "core/databasevf_schema.json"

@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., Awaitable[Any]]


def tool(description: str):
    """Enhanced decorator that properly handles async/sync functions"""
    def decorator(func: Callable):
        # Get function signature
        sig = inspect.signature(func)
        fields: Dict[str, tuple] = {}
        
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            ann = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else Any
            )
            default_value = (
                Field(default=param.default)
                if param.default is not inspect.Parameter.empty
                else ...
            )
            fields[pname] = (ann, default_value)
        
        # Create Pydantic model for arguments
        ArgsModel = create_model(f"{func.__name__.title()}Args", **fields)

        @wraps(func)
        async def _async_wrapper(*args, **kwargs):
            """Properly handle both sync and async functions"""
            print(f"[TOOL WRAPPER] Executing {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # Run sync function in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    if args and hasattr(args[0], '__class__'):  # Check if it's a method call
                        # For bound methods, we need to handle 'self' properly
                        result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                    else:
                        result = await loop.run_in_executor(None, partial(func, *args, **kwargs))
                
                print(f"[TOOL WRAPPER] {func.__name__} completed successfully")
                return result
            except Exception as e:
                print(f"[TOOL WRAPPER] Error in {func.__name__}: {e}")
                print(f"[TOOL WRAPPER] Traceback: {traceback.format_exc()}")
                raise

        # Attach tool specification - combine description and docstring
        combined_description = description
        if func.__doc__:
            # If we have both a description and docstring, combine them
            if description:
                combined_description = f"{description}\n\n{func.__doc__}"
            else:
                combined_description = func.__doc__
                
        _async_wrapper.__tool_spec__ = ToolSpec(
            name=func.__name__,
            description=combined_description or "No description",
            args_schema=ArgsModel,
            handler=_async_wrapper,
        )
        
        return _async_wrapper

    return decorator


class Tools:
    """Tools class to register tools and use them in the agent"""
    def __init__(self, db_path: str = "core/databasevf.db"):
        self.tools = []
        self.db_path = db_path

    

    def get_tools(self):
        return self.tools
    @tool("Returns 42")
    def dummy(self):
        return 42
    @tool("Do the sum of two numbers")
    def sum(self, a: int, b: int) -> int:
        return a + b

    @tool("Get the full database schema as a JSON object.")
    def get_db_schema(self) -> Dict[str, Any]:
        """Get the database schema"""
        try:
            if not os.path.exists(SCHEMA_PATH):
                return {"error": f"Schema file not found: {SCHEMA_PATH}"}
            
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load schema: {str(e)}"}

    @tool("Execute a read-only SQL query against the steel mill database.")
    def query_database(self, query: str) -> List[Dict]:
        """Runs a SQL query and returns a list of dictionaries."""
        try:
            if not os.path.exists(self.db_path):
                return [{"error": f"Database file not found: {self.db_path}"}]
            
            # Basic SQL injection prevention
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ['insert', 'update', 'delete', 'drop', 'create', 'alter']):
                return [{"error": "Only SELECT queries are allowed"}]
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(query)
            result = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return result
        except Exception as e:
            return [{"error": f"Query failed: {str(e)}"}]

    @tool("Get aggregated time-series data, formatted for a chart.")
    def get_timeseries_data(self, table_name: str, date_col: str, value_col: str, agg_func: str = "SUM") -> Dict[str, Any]:
        """Fetches daily aggregated data and formats it for a line chart."""
        try:
            # Validate aggregate function
            if agg_func.upper() not in ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX']:
                return {"error": "Invalid aggregate function. Use SUM, AVG, COUNT, MIN, or MAX"}
            
            query = f"""
                SELECT strftime('%Y-%m-%d', "{date_col}") as day, 
                       {agg_func}("{value_col}") as value
                FROM "{table_name}" 
                WHERE "{date_col}" IS NOT NULL 
                GROUP BY day 
                ORDER BY day
            """
            
            rows = self.query_database(query)
            
            if rows and isinstance(rows[0], dict) and "error" in rows[0]:
                return rows[0]
            
            return {
                "labels": [row["day"] for row in rows],
                "series": [{
                    "name": value_col.replace("_", " ").title(), 
                    "data": [row["value"] for row in rows]
                }]
            }
        except Exception as e:
            return {"error": f"Failed to get timeseries data: {str(e)}"}

    @tool("Calculates the total electrical consumption from EAF and LF values.")
    def calculate_total_consumption(self, cons_eaf_kwh: float, cons_lf_kwh: float) -> float:
        """Calculate total electrical consumption"""
        try:
            total = float(cons_eaf_kwh) + float(cons_lf_kwh)
            return total
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid input values: {str(e)}"}

    @tool("Assembles and generates a dashboard HTML file from a list of UI components.")
    def assemble_dashboard(self,
                        components: List[Dict[str, Any]],
                        title: str = "Steel Mill Production Dashboard",
                        outfile: str = "dashboard.html") -> str:
        """
        Takes a list of components and generates a dashboard HTML file.
        Returns the absolute path to the generated file.
        
        Supported component types:
        - heading: {"type": "heading", "text": "Title", "level": 1}
        - text/paragraph: {"type": "text", "text": "Some text"} or {"type": "paragraph", "text": "Some text"}
        - list: {"type": "list", "items": ["Item 1", "Item 2", "Item 3"]}
        - kpi: {"type": "kpi", "title": "KPI Title", "value": "123", "delta": 5}
        - chart: {"type": "chart", "title": "Chart Title", "labels": [...], "data": [...], "chart_type": "pie|line|bar"}
                or {"type": "chart", "title": "Chart Title", "data": {"labels": [...], "values": [...]}, "chart_type": "pie|line|bar"}
        - table: {"type": "table", "headers": [...], "rows": [[...]]}
        """
        try:
            # Create templates directory if it doesn't exist
            os.makedirs(TEMPLATES_DIR, exist_ok=True)
            
            # Debug: Print received components
            print(f"[DASHBOARD] Received {len(components)} components:")
            for i, comp in enumerate(components):
                print(f"[DASHBOARD]   {i}: {comp}")
            
            # Validate input
            if not isinstance(components, list):
                print(f"[DASHBOARD] ERROR: components is not a list, got {type(components)}: {components}")
                return f"Error: components must be a list, got {type(components)}"
            
            if len(components) == 0:
                print(f"[DASHBOARD] WARNING: Empty components list")
                return "Error: No components provided"
            
            # Transform components to the expected format
            transformed_components = []
            
            for comp in components:
                comp_type = comp.get("type", "")
                
                if comp_type == "heading":
                    # Create a heading component
                    level = comp.get("level", 1)
                    transformed_components.append({
                        "component": "HeadingComponent",
                        "props": {
                            "text": comp.get("text", ""),
                            "level": level
                        }
                    })
                    
                elif comp_type == "text" or comp_type == "paragraph":
                    # Create a text component (handle both "text" and "paragraph")
                    transformed_components.append({
                        "component": "TextComponent",
                        "props": {
                            "text": comp.get("text", "")
                        }
                    })
                    
                elif comp_type == "list":
                    # Create a list component by joining items
                    items = comp.get("items", [])
                    if isinstance(items, list):
                        list_html = "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"
                    else:
                        list_html = str(items)
                    transformed_components.append({
                        "component": "TextComponent",
                        "props": {
                            "text": list_html
                        }
                    })
                    
                elif comp_type == "kpi":
                    # Create a KPI component
                    transformed_components.append({
                        "component": "KPIBoxComponent",
                        "props": {
                            "title": comp.get("title", ""),
                            "value": str(comp.get("value", "")),
                            "delta": comp.get("delta")
                        }
                    })
                    
                elif comp_type == "chart":
                    # Create a chart component
                    chart_type = comp.get("chart_type", "line")
                    
                    # Handle different data structures
                    labels = []
                    data = []
                    
                    # Check if data is nested under a "data" key
                    if "data" in comp and isinstance(comp["data"], dict):
                        nested_data = comp["data"]
                        labels = nested_data.get("labels", [])
                        data = nested_data.get("values", nested_data.get("data", []))
                    else:
                        # Direct structure
                        labels = comp.get("labels", [])
                        data = comp.get("data", comp.get("values", []))
                    
                    if chart_type == "pie":
                        transformed_components.append({
                            "component": "PieChartComponent",
                            "props": {
                                "title": comp.get("title", ""),
                                "labels": labels,
                                "data": data
                            }
                        })
                    else:
                        # Default to line chart
                        if isinstance(data, list) and len(data) > 0 and not isinstance(data[0], dict):
                            # Simple data array, convert to series format
                            series = [{"name": "Data", "data": data}]
                        else:
                            # Assume it's already in series format
                            series = data if isinstance(data, list) else [{"name": "Data", "data": []}]
                        
                        transformed_components.append({
                            "component": "LineChartComponent",
                            "props": {
                                "title": comp.get("title", ""),
                                "labels": labels,
                                "series": series
                            }
                        })
                        
                elif comp_type == "table":
                    # Create a table component
                    transformed_components.append({
                        "component": "TableComponent",
                        "props": {
                            "headers": comp.get("headers", []),
                            "rows": comp.get("rows", [])
                        }
                    })
                else:
                    # Unknown component type, create a text component with error message
                    print(f"[DASHBOARD] Warning: Unknown component type '{comp_type}' in component: {comp}")
                    transformed_components.append({
                        "component": "TextComponent",
                        "props": {
                            "text": f"⚠️ Unsupported component type: {comp_type}"
                        }
                    })
            
            # Debug: Print transformed components
            print(f"[DASHBOARD] Transformed into {len(transformed_components)} components:")
            for i, comp in enumerate(transformed_components):
                print(f"[DASHBOARD]   {i}: {comp['component']} - {list(comp['props'].keys())}")
            
            # Validate we have components to render
            if len(transformed_components) == 0:
                print(f"[DASHBOARD] ERROR: No valid components after transformation")
                return "Error: No valid components found after transformation"
            
            # Create the HTML with the fixed template
            html = _FIXED_TEMPLATE_STR.format(
                page_title=title,
                components_json=json.dumps(transformed_components, ensure_ascii=False, indent=None)
            )
            
            # Debug: Print final JSON
            print(f"[DASHBOARD] Final components JSON: {json.dumps(transformed_components, indent=2)}")
            print(f"[DASHBOARD] HTML length: {len(html)} characters")
            
            # Write the output file
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(html)
            
            abs_path = os.path.abspath(outfile)
            return f"Dashboard successfully generated at: {abs_path}"
            
        except Exception as e:
            return f"Failed to generate dashboard: {str(e)}\nTraceback: {traceback.format_exc()}"

# --- Configuration ---
TEMPLATES_DIR = "dashboardgen/templates"
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)

# --- UI Component Helpers (Not tools, so no decorator) ---
def make_kpi(title: str, value: float, unit: str = "", delta: Optional[float] = None) -> Dict[str, Any]:
    """Create a KPI component"""
    return {
        "component": "KPIBoxComponent", 
        "props": {
            "title": title, 
            "value": f"{value:,.2f} {unit}".strip(), 
            "delta": delta
        }
    }

def make_line(labels: List[str], series: List[Dict[str, Any]], title: Optional[str] = None) -> Dict[str, Any]:
    """Create a line chart component"""
    return {
        "component": "LineChartComponent", 
        "props": {
            "title": title, 
            "labels": labels, 
            "series": series
        }
    }

def make_table(headers: List[str], rows: List[Tuple[Any, ...]]) -> Dict[str, Any]:
    """Create a table component"""
    return {
        "component": "TableComponent", 
        "props": {
            "headers": headers, 
            "rows": [list(r) for r in rows]
        }
    }


_FIXED_TEMPLATE_STR = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{page_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
      :root {{
        --bg: #f5f5f5; --fg: #333;
        --card-bg: #fff; --card-shadow: rgba(0,0,0,.1);
      }}
      [data-theme="dark"] {{
        --bg:#1e1e1e; --fg:#eee; --card-bg:#2a2a2a; --card-shadow: rgba(0,0,0,.6);
      }}
      body {{
        margin: 0;
        background: var(--bg);
        font: 16px "Segoe UI", Arial, sans-serif;
        color: var(--fg);
      }}
      #app {{
        display: grid;
        gap: 20px;
        padding: 25px;
        grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      }}
      .component-container {{
        background: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px var(--card-shadow);
      }}
      .kpi-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 150px;
      }}
      .kpi-value {{
        font-size: 2.7rem;
        font-weight: 700;
        color: #2196F3;
      }}
      .kpi-title {{
        margin-bottom: 10px;
        font-weight: 500;
        opacity: .8;
        text-align: center;
      }}
      .heading-container h1 {{
        margin: 0;
        color: #2196F3;
        text-align: center;
      }}
      .heading-container h2 {{
        margin: 0;
        color: #666;
        text-align: center;
      }}
      .heading-container h3 {{
        margin: 0;
        color: #888;
      }}
      .text-container {{
        font-size: 1.1rem;
        line-height: 1.6;
        text-align: center;
        padding: 10px 0;
      }}
      .text-container ul {{
        text-align: left;
        max-width: 400px;
        margin: 0 auto;
      }}
      .text-container li {{
        margin: 8px 0;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        padding: 8px 12px;
        border: 1px solid #ddd;
        text-align: left;
      }}
      th {{
        background: rgba(0,0,0,.05);
        font-weight: 600;
      }}
      .chart-container {{
        position: relative;
        height: 400px;
        width: 100%;
      }}
      .error-message {{
        color: red;
        font-weight: bold;
        text-align: center;
        padding: 20px;
      }}
    </style>
</head>
<body>
  <h1 style="text-align:center;margin:20px 0">{page_title}</h1>
  <div id="app"></div>
  
  <script>
    // Wait for Chart.js to load
    document.addEventListener('DOMContentLoaded', function() {{
      const components = {components_json};
      const root = document.getElementById('app');

      if (!root) {{
        console.error('Root element not found');
        return;
      }}

      console.log('Rendering components:', components);

      /* Component Library */
      const ComponentLibrary = {{
        HeadingComponent: {{
          render(el, props) {{
            el.className = (el.className || '') + ' heading-container';
            const level = props.level || 1;
            el.innerHTML = `<h${{level}}>${{props.text || 'Untitled'}}</h${{level}}>`;
          }}
        }},
        
        TextComponent: {{
          render(el, props) {{
            el.className = (el.className || '') + ' text-container';
            el.innerHTML = props.text || '';
          }}
        }},
        
        KPIBoxComponent: {{
          render(el, props) {{
            el.className = (el.className || '') + ' kpi-container';
            const deltaHtml = props.delta != null 
              ? ` <span style="font-size: 1rem; color: ${{props.delta > 0 ? 'green' : 'red'}}">
                  (${{props.delta > 0 ? '+' : ''}}${{props.delta}})
                 </span>` 
              : '';
            
            el.innerHTML = `
              <div class="kpi-title">${{props.title || 'KPI'}}</div>
              <div class="kpi-value">${{props.value || '0'}}${{deltaHtml}}</div>
            `;
          }}
        }},
        
        LineChartComponent: {{
          render(el, props) {{
            try {{
              if (typeof Chart === 'undefined') {{
                el.innerHTML = '<div class="error-message">Chart.js not loaded</div>';
                return;
              }}

              const chartContainer = document.createElement('div');
              chartContainer.className = 'chart-container';
              const canvas = document.createElement('canvas');
              chartContainer.appendChild(canvas);
              el.appendChild(chartContainer);
              
              console.log('Creating line chart with props:', props);
              
              const chartData = {{
                labels: props.labels || [],
                datasets: (props.series || []).map((s, i) => ({{
                  label: s.name || `Series ${{i + 1}}`,
                  data: s.data || [],
                  borderColor: `hsl(${{i * 137.5}}, 70%, 50%)`,
                  backgroundColor: `hsla(${{i * 137.5}}, 70%, 50%, .15)`,
                  tension: 0.35,
                  fill: false
                }}))
              }};
              
              const config = {{
                type: 'line',
                data: chartData,
                options: {{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {{
                    title: {{
                      display: !!(props.title),
                      text: props.title || ''
                    }},
                    legend: {{
                      display: true,
                      position: 'top'
                    }}
                  }},
                  scales: {{
                    y: {{
                      beginAtZero: true
                    }}
                  }}
                }}
              }};
              
              new Chart(canvas, config);
            }} catch (error) {{
              console.error('Error creating line chart:', error);
              el.innerHTML = `<div class="error-message">Error creating chart: ${{error.message}}</div>`;
            }}
          }}
        }},
        
        PieChartComponent: {{
          render(el, props) {{
            try {{
              if (typeof Chart === 'undefined') {{
                el.innerHTML = '<div class="error-message">Chart.js not loaded</div>';
                return;
              }}

              const chartContainer = document.createElement('div');
              chartContainer.className = 'chart-container';
              const canvas = document.createElement('canvas');
              chartContainer.appendChild(canvas);
              el.appendChild(chartContainer);
              
              console.log('Creating pie chart with props:', props);
              
              new Chart(canvas, {{
                type: 'pie',
                data: {{
                  labels: props.labels || [],
                  datasets: [{{
                    data: props.data || [],
                    backgroundColor: (props.labels || []).map((_, i) => `hsl(${{i * 360 / props.labels.length}}, 70%, 60%)`),
                    borderWidth: 2,
                    borderColor: '#fff'
                  }}]
                }},
                options: {{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {{
                    title: {{
                      display: !!(props.title),
                      text: props.title || ''
                    }},
                    legend: {{
                      position: 'bottom'
                    }}
                  }}
                }}
              }});
            }} catch (error) {{
              console.error('Error creating pie chart:', error);
              el.innerHTML = `<div class="error-message">Error creating chart: ${{error.message}}</div>`;
            }}
          }}
        }},
        
        TableComponent: {{
          render(el, props) {{
            try {{
              const table = document.createElement('table');
              
              // Create header
              const thead = document.createElement('thead');
              const headerRow = document.createElement('tr');
              (props.headers || []).forEach(header => {{
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
              }});
              thead.appendChild(headerRow);
              
              // Create body
              const tbody = document.createElement('tbody');
              (props.rows || []).forEach(row => {{
                const tr = document.createElement('tr');
                (row || []).forEach(cell => {{
                  const td = document.createElement('td');
                  td.textContent = cell;
                  tr.appendChild(td);
                }});
                tbody.appendChild(tr);
              }});
              
              table.appendChild(thead);
              table.appendChild(tbody);
              el.appendChild(table);
            }} catch (error) {{
              console.error('Error creating table:', error);
              el.innerHTML = `<div class="error-message">Error creating table: ${{error.message}}</div>`;
            }}
          }}
        }}
      }};

      /* Render components */
      components.forEach((component, index) => {{
        try {{
          const container = document.createElement('div');
          container.className = 'component-container';
          root.appendChild(container);
          
          const componentName = component.component;
          const props = component.props || {{}};
          
          console.log(`Rendering component ${{index}}: ${{componentName}}`, props);
          
          if (!componentName || !ComponentLibrary[componentName]) {{
            container.innerHTML = `<div class="error-message">❓ Unknown component: ${{componentName || 'undefined'}}</div>`;
            return;
          }}
          
          ComponentLibrary[componentName].render(container, props);
        }} catch (error) {{
          console.error(`Error rendering component ${{index}}:`, error);
          const container = document.createElement('div');
          container.className = 'component-container';
          container.innerHTML = `<div class="error-message">❌ Error rendering component: ${{error.message}}</div>`;
          root.appendChild(container);
        }}
      }});

      console.log('Dashboard rendering complete');
    }});
  </script>
</body>
</html>"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sympy import symbols, diff, parse_expr, sympify, solve
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.functions.elementary.trigonometric import TrigonometricFunction as trigonometric_functions
from sympy.functions.elementary.trigonometric import InverseTrigonometricFunction as inverse_trigonometric_functions
from sympy.utilities.lambdify import lambdify
import threading

# Helper functions for safe evaluation
def safe_eval(func, x_val, y_val=None):
    try:
        if y_val is None:
            result = func(x_val)
        else:
            result = func(x_val, y_val)
        # Check if result is a complex number
        if isinstance(result, complex):
            return np.nan
        return result
    except Exception:
        return np.nan

def lambdify_safe(symbols, expr):
    try:
        return lambdify(symbols, expr, "numpy")
    except Exception:
        # Fallback for complex expressions
        def safe_func(*args):
            try:
                subs_dict = dict(zip(symbols, args))
                return float(expr.subs(subs_dict))
            except Exception:
                return np.nan
        return safe_func

class GraphManager:
    def __init__(self, canvas_frame):
        self.figure = plt.figure(figsize=(8, 5), dpi=100, facecolor='#121212')
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax = None
        self.current_theme = 'dark'
        
    def init_2d_plot(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111, facecolor='#1E1E1E')
        self.apply_theme()
        self.canvas.draw()
        
    def init_3d_plot(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111, projection='3d', facecolor='#1E1E1E')
        self.apply_theme()
        self.canvas.draw()
        
    def apply_theme(self):
        if self.current_theme == 'dark':
            self.ax.set_facecolor('#1E1E1E')
            self.ax.tick_params(colors='#CCCCCC')
            self.ax.xaxis.label.set_color('#CCCCCC')
            self.ax.yaxis.label.set_color('#CCCCCC')
            if hasattr(self.ax, 'zaxis'):
                self.ax.zaxis.label.set_color('#CCCCCC')
            self.ax.title.set_color('#FFFFFF')
            
    def plot_2d(self, x, y, label, color='#33B5FF'):
        self.ax.plot(x, y, color, linewidth=2, label=label)
        self.ax.legend()
        self.canvas.draw()
        
    def plot_3d(self, X, Y, Z):
        self.ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        self.canvas.draw()

class DerivativeCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Derivadas Avanzada")
        self.root.geometry("1200x800")
        
        # Variables
        self.expr_var = tk.StringVar()
        self.order_var = tk.IntVar(value=1)
        self.x_min_var = tk.DoubleVar(value=-10)
        self.x_max_var = tk.DoubleVar(value=10)
        self.y_min_var = tk.DoubleVar(value=-10)
        self.y_max_var = tk.DoubleVar(value=10)
        self.is_3d_var = tk.BooleanVar(value=False)
        self.current_theme = 'dark'
        self.history = []
        
        # Configuración inicial
        self.configure_dark_theme()
        self.create_menu()
        self.create_widgets()
        self.setup_shortcuts()
        self.show_splash()
        
        # Ejemplos de funciones
        self.examples = {
            "Polinomios": [
                "x^2 + 3*x - 5",
                "2*x^3 - 4*x^2 + x - 7",
                "x^5 - 7*x^3 + 2*x",
                "3*x^4 - 2*x^3 + 5*x^2 - x + 1",
                "x^6 - 6*x^5 + 15*x^4 - 20*x^3",
                "7*x^8 - 3*x^6 + 2*x^4 - x^2 + 8",
                "x^10 - 10*x^9 + 45*x^8 - 120*x^7",
                "2*x^3 + 3*x^2 - 5*x + 7",
                "4*x^7 - 3*x^5 + 2*x^3 - x",
                "x^12 - 12*x^10 + 66*x^8 - 220*x^6",
                "5*x^4 - 4*x^3 + 3*x^2 - 2*x + 1",
                "x^15 - 15*x^12 + 105*x^9 - 455*x^6 + 1365*x^3 - 3003"
            ],
            "Trigonométricas": [
                "sin(x)",
                "cos(2*x)",
                "tan(x/2)",
                "sin(x)*cos(x)",
                "sin(x^2)",
                "sin(x) + cos(x)",
                "sin(x)^2",
                "sin(x)/cos(x)",
                "sin(x)*tan(x)",
                "cos(x)^2 - sin(x)^2",
                "sin(3*x)",
                "cos(x/3)",
                "sin(x)*cos(2*x)",
                "tan(x)^2",
                "sin(x)*sin(2*x)"
            ],
            "Trig. Inversas": [
                "asin(x)",
                "acos(x)",
                "atan(x)",
                "acot(x)",
                "asec(x)",
                "acsc(x)",
                "asin(x^2)",
                "acos(2*x)",
                "atan(x/2)",
                "asin(sqrt(x))",
                "acos(1/x)",
                "atan(exp(x))",
                "asin(x) + acos(x)",
                "atan(x) - acot(x)",
                "asec(x^2) + acsc(x^2)"
            ],
            "Exponenciales": [
                "exp(x)",
                "exp(-x^2)",
                "2^x",
                "x*exp(x)",
                "exp(sin(x))",
                "exp(x^2)",
                "3^x",
                "exp(x)/x",
                "x^2*exp(x)",
                "exp(x)*sin(x)",
                "exp(-x)",
                "exp(x^3)",
                "x*exp(-x^2)",
                "exp(cos(x))"
            ],
            "Logarítmicas": [
                "ln(x)",
                "log(x, 10)",
                "x*ln(x)",
                "ln(x^2 + 1)",
                "ln(sin(x))",
                "ln(x^2)",
                "ln(1+x)/x",
                "log(x^3, 10)",
                "x*log(x, 2)",
                "ln(exp(x))",
                "ln(1/x)",
                "ln(x+1)^2",
                "ln(cos(x))",
                "log(x, 3)"
            ],
            "Raíces": [
                "sqrt(x)",
                "x*sqrt(x)",
                "sqrt(1 - x^2)",
                "sqrt(x^2 + 1)",
                "sqrt(sin(x))",
                "x^2*sqrt(x)",
                "sqrt(x)/x",
                "sqrt(x^3)",
                "sqrt(x+1)",
                "sqrt(x^2 - 1)",
                "sqrt(cos(x))",
                "(x+1)*sqrt(x)",
                "sqrt(x*ln(x))"
            ],
            "Racionales": [
                "1/x",
                "(x^2 + 1)/(x - 1)",
                "x/(x^2 + 4)",
                "1/(x^2)",
                "(x+1)/(x-1)",
                "x^2/(x^3+1)",
                "(x^3 - 3*x)/(x^2 - 4)",
                "x/(x^2 - 1)",
                "(sin(x))/(cos(x))",
                "(x^2 - 1)/(x^2 + 1)",
                "1/(x*(x+1))",
                "(x^3)/(x^4 + 1)",
                "(x-1)/(x^2+x+1)"
            ],
            "Compuestas": [
                "sin(exp(x))",
                "ln(sin(x))",
                "exp(sin(x^2))",
                "sin(ln(x))",
                "sqrt(exp(x))",
                "ln(cos(x^2))",
                "exp(ln(x))",
                "cos(sin(x))",
                "tan(ln(x))",
                "ln(sqrt(x))",
                "sqrt(sin(x)^2)",
                "exp(cos(x)^2)",
                "sin(x^3)*cos(x^2)",
                "ln(exp(sin(x)))"
            ],
            "Operaciones Especiales": [
                "x^x",
                "(sin(x))^(cos(x))",
                "x^(1/x)",
                "e^(x*ln(x))",
                "x*sin(x)",
                "x^2*cos(x)",
                "exp(x)*ln(x)",
                "sin(x)*cos(x)*tan(x)",
                "x*sin(x)*exp(x)",
                "sqrt(x)*ln(x)"
            ],
            "Funciones 3D": [
                "x + y",
                "x^2 + y^2",
                "sin(x) + cos(y)",
                "x*y",
                "x^2 - y^2",
                "sin(x^2 + y^2)",
                "exp(-(x^2 + y^2))",
                "x*sin(y) + y*sin(x)",
                "sqrt(x^2 + y^2)",
                "sin(x)*cos(y)",
                "x^3 + y^3",
                "log(x^2 + y^2 + 1)",
                "tan(x*y)"
            ]
        }
    
    def configure_dark_theme(self):
        style = ttk.Style()
        style.theme_use('alt')
        
        style.configure('.', background='#121212', foreground='#FFFFFF')
        style.configure('TFrame', background='#121212')
        style.configure('TLabel', background='#121212', foreground='#FFFFFF')
        style.configure('TButton', background='#2a2a2a', foreground='#FFFFFF')
        style.configure('TEntry', fieldbackground='#2a2a2a', foreground='#FFFFFF')
        style.configure('TLabelframe', background='#121212', foreground='#FFFFFF')
        style.configure('TLabelframe.Label', background='#121212', foreground='#33B5FF')
        style.configure('TNotebook', background='#121212')
        style.configure('TNotebook.Tab', background='#2a2a2a', foreground='#FFFFFF')
        style.map('TNotebook.Tab', background=[('selected', '#33B5FF')], foreground=[('selected', '#000000')])
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Nuevo", command=self.clear_input, accelerator="Ctrl+N")
        file_menu.add_command(label="Exportar resultados", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        
        # Menú Ver
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Tema Oscuro", command=lambda: self.change_theme('dark'))
        view_menu.add_command(label="Tema Claro", command=lambda: self.change_theme('light'))
        menubar.add_cascade(label="Ver", menu=view_menu)
        
        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Instrucciones", command=self.show_help)
        help_menu.add_command(label="Acerca de", command=self.show_about)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(main_frame, text="Calculadora de Derivadas Avanzada", 
                 font=("Arial", 18, "bold"), foreground="#33B5FF").pack(pady=10)
        
        # Frame de entrada
        input_frame = ttk.LabelFrame(main_frame, text="Entrada", padding=10)
        input_frame.pack(fill=tk.X, pady=10)
        
        # Función
        expr_frame = ttk.Frame(input_frame)
        expr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(expr_frame, text="Función f(x,y):", width=15).pack(side=tk.LEFT)
        expr_entry = ttk.Entry(expr_frame, textvariable=self.expr_var, width=40)
        expr_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(expr_frame, text="Ejemplos", command=self.show_examples).pack(side=tk.LEFT, padx=5)
        
        # Botones especiales
        special_ops_frame = ttk.Frame(input_frame)
        special_ops_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(special_ops_frame, text="Potenciación (a^b)", 
                  command=lambda: self.insert_operation("^")).pack(side=tk.LEFT, padx=2)
        ttk.Button(special_ops_frame, text="Producto (a*b)", 
                  command=lambda: self.insert_operation("*")).pack(side=tk.LEFT, padx=2)
        
        # Checkbox para 3D
        ttk.Checkbutton(special_ops_frame, text="Gráfica 3D", 
                       variable=self.is_3d_var, command=self.toggle_3d).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(special_ops_frame, 
                 text="Nota: Para productos use '*' explícitamente (ej: x*sin(x))",
                 foreground="#FFA500", font=("Arial", 9)).pack(side=tk.LEFT, padx=10)
        
        # Orden de derivada
        order_frame = ttk.Frame(input_frame)
        order_frame.pack(fill=tk.X, pady=5)
        ttk.Label(order_frame, text="Orden de derivada:", width=15).pack(side=tk.LEFT)
        ttk.Spinbox(order_frame, from_=1, to=10, textvariable=self.order_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Rangos
        range_frame = ttk.Frame(input_frame)
        range_frame.pack(fill=tk.X, pady=5)
        ttk.Label(range_frame, text="Rango de gráfica:", width=15).pack(side=tk.LEFT)
        ttk.Label(range_frame, text="x mín:").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.x_min_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(range_frame, text="x máx:").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.x_max_var, width=5).pack(side=tk.LEFT, padx=5)
        
        y_range_frame = ttk.Frame(input_frame)
        y_range_frame.pack(fill=tk.X, pady=5)
        ttk.Label(y_range_frame, text="", width=15).pack(side=tk.LEFT)
        ttk.Label(y_range_frame, text="y mín:").pack(side=tk.LEFT)
        ttk.Entry(y_range_frame, textvariable=self.y_min_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(y_range_frame, text="y máx:").pack(side=tk.LEFT)
        ttk.Entry(y_range_frame, textvariable=self.y_max_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Botones principales
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Calcular Derivada", command=self.calculate_derivative_threaded).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calcular Máximos/Mínimos", command=self.find_extrema).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Limpiar", command=self.clear_input).pack(side=tk.LEFT, padx=5)
        
        # Notebook para resultados
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Pestaña de resultados
        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="Resultados")
        
        # Resultados
        expr_result_frame = ttk.Frame(results_tab)
        expr_result_frame.pack(fill=tk.X, pady=5)
        ttk.Label(expr_result_frame, text="Función Original:", width=15).pack(side=tk.LEFT)
        self.original_expr = ttk.Label(expr_result_frame, text="", width=50, foreground="#FFFFFF")
        self.original_expr.pack(side=tk.LEFT, padx=5)
        
        derivative_result_frame = ttk.Frame(results_tab)
        derivative_result_frame.pack(fill=tk.X, pady=5)
        ttk.Label(derivative_result_frame, text="Derivada:", width=15).pack(side=tk.LEFT)
        self.derivative_expr = ttk.Label(derivative_result_frame, text="", width=50, foreground="#FFFFFF")
        self.derivative_expr.pack(side=tk.LEFT, padx=5)
        
        extrema_result_frame = ttk.Frame(results_tab)
        extrema_result_frame.pack(fill=tk.X, pady=5)
        ttk.Label(extrema_result_frame, text="Extremos:", width=15).pack(side=tk.LEFT)
        self.extrema_expr = ttk.Label(extrema_result_frame, text="", width=50, foreground="#FFFFFF")
        self.extrema_expr.pack(side=tk.LEFT, padx=5)
        
        # Pestaña de gráficos
        graph_tab = ttk.Frame(notebook)
        notebook.add(graph_tab, text="Gráficos")
        
        # Gráfica
        self.graph_manager = GraphManager(graph_tab)
        self.graph_manager.init_2d_plot()
        
        # Historial
        history_tab = ttk.Frame(notebook)
        notebook.add(history_tab, text="Historial")
        
        self.history_listbox = tk.Listbox(history_tab, bg='#2a2a2a', fg='#FFFFFF', selectbackground='#33B5FF')
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.history_listbox.bind('<<ListboxSelect>>', self.use_history_item)
        
        clear_history_btn = ttk.Button(history_tab, text="Limpiar Historial", command=self.clear_history)
        clear_history_btn.pack(pady=5)
    
    def setup_shortcuts(self):
        self.root.bind('<Control-n>', lambda e: self.clear_input())
        self.root.bind('<Control-d>', lambda e: self.calculate_derivative_threaded())
        self.root.bind('<Control-l>', lambda e: self.clear_input())
        self.root.bind('<Control-e>', lambda e: self.export_results())
        self.root.bind('<F1>', lambda e: self.show_help())
    
    def show_splash(self):
        splash = tk.Toplevel(self.root)
        splash.title("Cargando...")
        splash.geometry("300x200")
        splash.overrideredirect(True)
        
        # Centrar en pantalla
        w, h = 300, 200
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        splash.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        ttk.Label(splash, text="Calculadora de Derivadas", font=("Arial", 14)).pack(pady=20)
        ttk.Label(splash, text="Cargando...").pack()
        
        self.root.after(1500, splash.destroy)
    
    def toggle_3d(self):
        if self.is_3d_var.get():
            self.graph_manager.init_3d_plot()
        else:
            self.graph_manager.init_2d_plot()
        
        if hasattr(self, 'current_expr'):
            self.plot_functions(self.current_expr, getattr(self, 'current_derivative', None))
    
    def insert_operation(self, op):
        current = self.expr_var.get()
        try:
            cursor_pos = self.root.focus_get().index(tk.INSERT)
            self.expr_var.set(current[:cursor_pos] + op + current[cursor_pos:])
            self.root.focus_get().focus_set()
            self.root.focus_get().icursor(cursor_pos + len(op))
        except Exception:
            self.expr_var.set(current + op)
    
    def validate_input(self):
        expr = self.expr_var.get().strip()
        if not expr:
            messagebox.showerror("Error", "La función no puede estar vacía")
            return False
        
        try:
            parse_expr(expr.replace("^", "**"))
            return True
        except Exception as e:
            messagebox.showerror("Error de sintaxis", f"Error en la función:\n{str(e)}")
            return False
    
    def calculate_derivative_threaded(self):
        if not self.validate_input():
            return
        
        # Mostrar indicador de carga
        self.calculating_label = ttk.Label(self.root, text="Calculando...", foreground="#33B5FF")
        self.calculating_label.place(relx=0.5, rely=0.5, anchor='center')
        self.root.update()
        
        threading.Thread(target=self.calculate_derivative, daemon=True).start()
    
    def calculate_derivative(self):
        try:
            expr_str = self.expr_var.get()
            order = self.order_var.get()
            
            expr_str = expr_str.replace("^", "**")
            is_3d = self.is_3d_var.get()
            
            if is_3d:
                x, y = symbols('x y')
                variables = (x, y)
            else:
                x = symbols('x')
                variables = (x,)
            
            expr = parse_expr(expr_str)
            
            derivada = expr
            for i in range(order):
                derivada = diff(derivada, variables[0])
            
            self.original_expr.config(text=str(expr).replace("**", "^"))
            self.derivative_expr.config(text=str(derivada).replace("**", "^"))
            
            if is_3d:
                self.plot_3d_function(expr)
            else:
                self.plot_functions(expr, derivada)
            
            self.show_derivative_details(expr, derivada, order)
            
            self.current_expr = expr
            self.current_derivative = derivada
            self.add_to_history(expr_str)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")
        finally:
            # Ocultar indicador de carga
            self.calculating_label.place_forget()
    
    def plot_3d_function(self, expr):
        try:
            x_min = self.x_min_var.get()
            x_max = self.x_max_var.get()
            y_min = self.y_min_var.get()
            y_max = self.y_max_var.get()
            
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            
            x_sym, y_sym = symbols('x y')
            f = lambdify_safe((x_sym, y_sym), expr)
            
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i,j] = safe_eval(f, X[i,j], Y[i,j])
            
            self.graph_manager.ax.clear()
            surf = self.graph_manager.ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            
            self.graph_manager.ax.set_xlabel('x', color='white')
            self.graph_manager.ax.set_ylabel('y', color='white')
            self.graph_manager.ax.set_zlabel('z', color='white')
            
            self.graph_manager.ax.set_title('Gráfica 3D de la función', color='white')
            self.graph_manager.figure.colorbar(surf, ax=self.graph_manager.ax, pad=0.1)
            
            self.graph_manager.canvas.draw()
            
        except Exception as e:
            print(f"Error al graficar 3D: {e}")
            messagebox.showerror("Error", f"No se pudo graficar la función 3D: {e}")
    
    def plot_functions(self, expr, derivada=None):
        try:
            self.graph_manager.ax.clear()
            
            if self.is_3d_var.get():
                self.plot_3d_function(expr)
                return
            
            x_min = self.x_min_var.get()
            x_max = self.x_max_var.get()
            
            x = symbols('x')
            f = lambdify_safe((x,), expr)
            
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = [safe_eval(f, x_val) for x_val in x_vals]
            
            valid_points = [(x, y) for x, y in zip(x_vals, y_vals) if np.isfinite(y) and not np.isnan(y)]
            
            if valid_points:
                x_valid, y_valid = zip(*valid_points)
                self.graph_manager.ax.plot(x_valid, y_valid, '#33B5FF', linewidth=2, label='f(x)')
            
            if derivada is not None:
                df = lambdify_safe((x,), derivada)
                dy_vals = [safe_eval(df, x_val) for x_val in x_vals]
                
                valid_derivs = [(x, y) for x, y in zip(x_vals, dy_vals) if np.isfinite(y) and not np.isnan(y)]
                
                if valid_derivs:
                    x_deriv, y_deriv = zip(*valid_derivs)
                    self.graph_manager.ax.plot(x_deriv, y_deriv, '#FF5733', linewidth=2, label='f\'(x)')
            
            self.graph_manager.ax.axhline(y=0, color='#555555', linestyle='-', alpha=0.5)
            self.graph_manager.ax.axvline(x=0, color='#555555', linestyle='-', alpha=0.5)
            self.graph_manager.ax.grid(True, color='#555555', alpha=0.3)
            
            self.graph_manager.ax.set_xlabel('x', fontsize=10)
            self.graph_manager.ax.set_ylabel('y', fontsize=10)
            self.graph_manager.ax.set_title('Función y su Derivada' if derivada is not None else 'Función', fontsize=12)
            
            handles, labels = self.graph_manager.ax.get_legend_handles_labels()
            if handles:
                legend = self.graph_manager.ax.legend(handles, labels)
                frame = legend.get_frame()
                frame.set_facecolor('#2a2a2a')
                frame.set_edgecolor('#555555')
                for text in legend.get_texts():
                    text.set_color('#FFFFFF')
            
            self.graph_manager.canvas.draw()
            
        except Exception as e:
            print(f"Error al graficar: {e}")
    
    def find_extrema(self):
        try:
            if not hasattr(self, 'current_expr'):
                messagebox.showinfo("Información", "Primero calcule la derivada de una función.")
                return
            
            x = symbols('x')
            primera_derivada = diff(self.current_expr, x)
            segunda_derivada = diff(primera_derivada, x)
            
            try:
                critical_points = solve(primera_derivada, x)
            except Exception:
                messagebox.showinfo("Información", "No se pueden determinar analíticamente los puntos críticos.")
                self.extrema_expr.config(text="No se pueden calcular analíticamente")
                return
            
            if not critical_points:
                self.extrema_expr.config(text="No se encontraron puntos críticos en el dominio real")
                return
            
            maxima = []
            minima = []
            inflection = []
            undetermined = []
            
            for point in critical_points:
                try:
                    if hasattr(point, 'is_real') and not point.is_real:
                        continue
                    
                    second_deriv_value = segunda_derivada.subs(x, point)
                    
                    if second_deriv_value < 0:
                        maxima.append(point)
                    elif second_deriv_value > 0:
                        minima.append(point)
                    elif second_deriv_value == 0:
                        inflection.append(point)
                    else:
                        undetermined.append(point)
                except Exception:
                    undetermined.append(point)
            
            result_text = ""
            if maxima:
                result_text += f"Máximos en x = {', '.join([str(p) for p in maxima])}\n"
            if minima:
                result_text += f"Mínimos en x = {', '.join([str(p) for p in minima])}\n"
            if inflection:
                result_text += f"Posibles puntos de inflexión en x = {', '.join([str(p) for p in inflection])}\n"
            if undetermined:
                result_text += f"Puntos indeterminados en x = {', '.join([str(p) for p in undetermined])}\n"
            
            if not result_text:
                result_text = "No se encontraron extremos en el dominio real"
            
            self.extrema_expr.config(text=result_text)
            self.plot_with_extrema(self.current_expr, maxima, minima, inflection)
            self.show_extrema_details(self.current_expr, maxima, minima, inflection, undetermined)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular extremos: {e}")
    
    def plot_with_extrema(self, expr, maxima, minima, inflection):
        try:
            self.graph_manager.ax.clear()
            
            x_min = self.x_min_var.get()
            x_max = self.x_max_var.get()
            
            x = symbols('x')
            f = lambdify_safe((x,), expr)
            
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = [safe_eval(f, x_val) for x_val in x_vals]
            
            valid_points = [(x, y) for x, y in zip(x_vals, y_vals) if np.isfinite(y) and not np.isnan(y)]
            
            if valid_points:
                x_valid, y_valid = zip(*valid_points)
                self.graph_manager.ax.plot(x_valid, y_valid, '#33B5FF', linewidth=2, label='f(x)')
            
            max_style = {'color': 'green', 'marker': '^', 's': 100, 'label': 'Máximo'}
            min_style = {'color': 'red', 'marker': 'v', 's': 100, 'label': 'Mínimo'}
            inf_style = {'color': 'yellow', 'marker': 'o', 's': 80, 'label': 'Inflexión'}
            
            def get_annotation_position(index, total, x_pos, y_pos):
                base_offset = 0.8
                angle = (index / total) * 2 * np.pi
                radius = base_offset * (1 + index % 3 * 0.3)
                return (x_pos + radius * np.cos(angle), y_pos + radius * np.sin(angle))
            
            # Procesar máximos, mínimos y puntos de inflexión
            for i, point in enumerate(maxima):
                if x_min <= float(point) <= x_max:
                    try:
                        y_value = float(f(float(point)))
                        if np.isfinite(y_value) and not np.isnan(y_value):
                            self.graph_manager.ax.scatter([float(point)], [y_value], **max_style)
                            text_x, text_y = get_annotation_position(i, len(maxima), float(point), y_value)
                            self.graph_manager.ax.annotate(f'Máx ({float(point):.2f}, {y_value:.2f})',
                                                          xy=(float(point), y_value),
                                                          xytext=(text_x, text_y),
                                                          color='white',
                                                          bbox=dict(boxstyle='round,pad=0.3', fc='green', alpha=0.7),
                                                          arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7))
                            max_style['label'] = None
                    except Exception as e:
                        print(f"Error al marcar máximo: {e}")
            
            for i, point in enumerate(minima):
                if x_min <= float(point) <= x_max:
                    try:
                        y_value = float(f(float(point)))
                        if np.isfinite(y_value) and not np.isnan(y_value):
                            self.graph_manager.ax.scatter([float(point)], [y_value], **min_style)
                            text_x, text_y = get_annotation_position(i + 0.5, len(minima), float(point), y_value)
                            self.graph_manager.ax.annotate(f'Mín ({float(point):.2f}, {y_value:.2f})',
                                                          xy=(float(point), y_value),
                                                          xytext=(text_x, text_y),
                                                          color='white',
                                                          bbox=dict(boxstyle='round,pad=0.3', fc='red', alpha=0.7),
                                                          arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7))
                            min_style['label'] = None
                    except Exception as e:
                        print(f"Error al marcar mínimo: {e}")
            
            for i, point in enumerate(inflection):
                if x_min <= float(point) <= x_max:
                    try:
                        y_value = float(f(float(point)))
                        if np.isfinite(y_value) and not np.isnan(y_value):
                            self.graph_manager.ax.scatter([float(point)], [y_value], **inf_style)
                            text_x, text_y = get_annotation_position(i + 0.3, len(inflection), float(point), y_value)
                            self.graph_manager.ax.annotate(f'Inf ({float(point):.2f}, {y_value:.2f})',
                                                          xy=(float(point), y_value),
                                                          xytext=(text_x, text_y),
                                                          color='black',
                                                          bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                                                          arrowprops=dict(facecolor='yellow', shrink=0.05, alpha=0.7))
                            inf_style['label'] = None
                    except Exception as e:
                        print(f"Error al marcar punto de inflexión: {e}")
            
            self.graph_manager.ax.set_facecolor('#1E1E1E')
            self.graph_manager.ax.axhline(y=0, color='#555555', linestyle='-', alpha=0.5)
            self.graph_manager.ax.axvline(x=0, color='#555555', linestyle='-', alpha=0.5)
            self.graph_manager.ax.grid(True, color='#555555', alpha=0.3)
            
            self.graph_manager.ax.set_xlabel('x', fontsize=10)
            self.graph_manager.ax.set_ylabel('y', fontsize=10)
            self.graph_manager.ax.set_title('Función con Extremos y Puntos de Inflexión', fontsize=12)
            
            handles, labels = self.graph_manager.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                legend = self.graph_manager.ax.legend(by_label.values(), by_label.keys())
                frame = legend.get_frame()
                frame.set_facecolor('#2a2a2a')
                frame.set_edgecolor('#555555')
                for text in legend.get_texts():
                    text.set_color('#FFFFFF')
            
            self.graph_manager.ax.autoscale_view()
            self.graph_manager.canvas.draw()
            
        except Exception as e:
            print(f"Error al graficar: {e}")
    
    def show_derivative_details(self, expr, derivada, order):
        popup = tk.Toplevel(self.root)
        popup.title("Detalles de la Derivada")
        popup.geometry("700x600")
        popup.configure(bg="#121212")
        popup.grab_set()
        
        popup_frame = ttk.Frame(popup, padding=15)
        popup_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(popup_frame, text="Resultado del Cálculo de Derivada", 
                  font=("Arial", 14, "bold"), foreground="#33B5FF").pack(pady=10)
        
        expr_display = str(expr).replace('**', '^')
        derivada_display = str(derivada).replace('**', '^')
        
        original_frame = ttk.Frame(popup_frame)
        original_frame.pack(fill=tk.X, pady=5)
        ttk.Label(original_frame, text="Función Original:", 
                  font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        ttk.Label(original_frame, text=expr_display, 
                  font=("Arial", 12), foreground="#FFFFFF").pack(side=tk.LEFT, padx=5)
        
        derivative_frame = ttk.Frame(popup_frame)
        derivative_frame.pack(fill=tk.X, pady=5)
        orden_texto = "" if order == 1 else f"de orden {order}"
        ttk.Label(derivative_frame, text=f"Derivada {orden_texto}:", 
                  font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        ttk.Label(derivative_frame, text=derivada_display, 
                  font=("Arial", 12), foreground="#FFFFFF").pack(side=tk.LEFT, padx=5)
        
        procedure_frame = ttk.LabelFrame(popup_frame, text="Procedimiento Paso a Paso", padding=10)
        procedure_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        procedure_canvas = tk.Canvas(procedure_frame, highlightthickness=0, bg="#1E1E1E")
        procedure_scrollbar = ttk.Scrollbar(procedure_frame, orient="vertical", command=procedure_canvas.yview)
        procedure_subframe = ttk.Frame(procedure_canvas)

        procedure_canvas.configure(yscrollcommand=procedure_scrollbar.set)
        procedure_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        procedure_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        procedure_canvas.create_window((0, 0), window=procedure_subframe, anchor="nw")
        procedure_subframe.bind("<Configure>", lambda e, c=procedure_canvas: c.configure(scrollregion=c.bbox("all")))

        steps = self.generate_derivative_steps(expr, order)
        for i, step in enumerate(steps):
            step_frame = ttk.Frame(procedure_subframe)
            step_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(step_frame, text=f"Paso {i+1}:", 
                    font=("Arial", 10, "bold"), foreground="#33B5FF").pack(side=tk.LEFT, padx=5)
            
            ttk.Label(step_frame, text=step["description"], 
                    font=("Arial", 10), foreground="#FFFFFF").pack(side=tk.LEFT, padx=5)
            
            if "result" in step:
                result_frame = ttk.Frame(procedure_subframe)
                result_frame.pack(fill=tk.X, pady=2, padx=20)
                
                result_text = step["result"].replace('**', '^')
                ttk.Label(result_frame, text=result_text, 
                        font=("Arial", 10, "italic"), foreground="#CCCCCC").pack(side=tk.LEFT)
        
        info_text = f"Se calculó la derivada {'de primer orden' if order == 1 else f'de orden {order}'} de la función."
        
        expr_str = str(expr_display).lower()
        if "/" in expr_str:
            info_text += "\nLa función contiene divisiones, se aplicó la regla del cociente."
        if "sqrt" in expr_str or "^(1/" in expr_str:
            info_text += "\nLa función contiene raíces, se aplicó la regla de la cadena."
        if "sin" in expr_str or "cos" in expr_str or "tan" in expr_str:
            info_text += "\nLa función contiene trigonométricas, se aplicaron las derivadas correspondientes."
        if "asin" in expr_str or "acos" in expr_str or "atan" in expr_str or "acot" in expr_str or "asec" in expr_str or "acsc" in expr_str:
            info_text += "\nLa función contiene trigonométricas inversas, se aplicaron las derivadas específicas."
        if "exp" in expr_str:
            info_text += "\nLa función contiene exponenciales, se aplicó la propiedad de que la derivada de e^x es e^x."
        if "ln" in expr_str or "log" in expr_str:
            info_text += "\nLa función contiene logaritmos, se aplicó la regla de derivación logarítmica."
        
        ttk.Label(popup_frame, text=info_text, 
                font=("Arial", 10, "italic"), foreground="#CCCCCC", wraplength=450).pack(pady=10)
        
        ttk.Button(popup_frame, text="Cerrar", command=popup.destroy).pack(pady=10)
    
    def generate_derivative_steps(self, expr, order):
        steps = []
        x = symbols('x')
        expr_sympy = expr
        current_expr = expr_sympy
        
        steps.append({
            "description": "Función original a derivar:",
            "result": str(current_expr)
        })
        
        if order > 1:
            for i in range(1, order+1):
                if i == 1:
                    description = "Calculando la primera derivada:"
                else:
                    description = f"Calculando la derivada {i}-ésima:"
                
                expr_str = str(current_expr).lower()
                rule_explanation = self.identify_derivative_rule(expr_str)
                
                if rule_explanation:
                    description += f" {rule_explanation}"
                
                current_expr = diff(current_expr, x)
                
                steps.append({
                    "description": description,
                    "result": str(current_expr)
                })
        else:
            expr_str = str(expr_sympy).lower()
            components = self.decompose_expression(expr_sympy)
            
            for component, explanation in components:
                steps.append({
                    "description": f"Aplicando {explanation}:",
                    "result": f"d/dx({component}) = {str(diff(parse_expr(str(component)), x))}"
                })
            
            if len(components) > 1:
                steps.append({
                    "description": "Combinando todos los términos:",
                    "result": str(diff(expr_sympy, x))
                })
        
        return steps

    def identify_derivative_rule(self, expr_str):
        if "+" in expr_str or "-" in expr_str:
            return "usando la regla de la suma/resta"
        elif "*" in expr_str:
            return "usando la regla del producto"
        elif "/" in expr_str:
            return "usando la regla del cociente"
        elif "^" in expr_str or "**" in expr_str:
            return "usando la regla de la potencia"
        elif any(trig in expr_str for trig in ["sin", "cos", "tan"]):
            return "usando las fórmulas de derivación trigonométrica"
        elif any(inv_trig in expr_str for inv_trig in ["asin", "acos", "atan", "acot", "asec", "acsc"]):
            return "usando las fórmulas de derivación de funciones trigonométricas inversas"
        elif "exp" in expr_str:
            return "usando que la derivada de e^x es e^x"
        elif "ln" in expr_str or "log" in expr_str:
            return "usando la regla de derivación logarítmica"
        elif "sqrt" in expr_str:
            return "usando la regla de la cadena para la raíz cuadrada"
        else:
            return ""

    def decompose_expression(self, expr):
        components = []
        x = symbols('x')
        
        if isinstance(expr, Add):
            for arg in expr.args:
                components.append((arg, self.get_component_explanation(arg)))
        else:
            components.append((expr, self.get_component_explanation(expr)))
        
        return components

    def get_component_explanation(self, expr):
        expr_str = str(expr).lower()
        
        if isinstance(expr, Pow):
            return "regla de la potencia"
        elif isinstance(expr, Mul):
            return "regla del producto"
        elif isinstance(expr, trigonometric_functions):
            return "derivada de función trigonométrica"
        elif isinstance(expr, inverse_trigonometric_functions):
            return "derivada de función trigonométrica inversa"
        elif "exp" in expr_str:
            return "derivada de la función exponencial"
        elif "log" in expr_str or "ln" in expr_str:
            return "derivada de la función logarítmica"
        elif "sqrt" in expr_str:
            return "derivada de la raíz cuadrada"
        else:
            return "regla básica de derivación"
    
    def show_extrema_details(self, expr, maxima, minima, inflection, undetermined):
        popup = tk.Toplevel(self.root)
        popup.title("Detalles de Máximos y Mínimos")
        popup.geometry("700x600")
        popup.configure(bg="#121212")
        popup.grab_set()
        
        popup_frame = ttk.Frame(popup, padding=15)
        popup_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(popup_frame, text="Análisis de Extremos", 
                  font=("Arial", 14, "bold"), foreground="#33B5FF").pack(pady=10)
        
        expr_display = str(expr).replace('**', '^')
        
        original_frame = ttk.Frame(popup_frame)
        original_frame.pack(fill=tk.X, pady=5)
        ttk.Label(original_frame, text="Función Analizada:", 
                  font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        ttk.Label(original_frame, text=expr_display, 
                  font=("Arial", 12), foreground="#FFFFFF").pack(side=tk.LEFT, padx=5)
        
        details_frame = ttk.LabelFrame(popup_frame, text="Resultados del Análisis", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        details_canvas = tk.Canvas(details_frame, highlightthickness=0, bg="#1E1E1E")
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=details_canvas.yview)
        details_subframe = ttk.Frame(details_canvas)
        
        details_canvas.configure(yscrollcommand=details_scrollbar.set)
        details_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        details_canvas.create_window((0, 0), window=details_subframe, anchor="nw")
        details_subframe.bind("<Configure>", lambda e, c=details_canvas: c.configure(scrollregion=c.bbox("all")))
        
        if maxima:
            max_frame = ttk.Frame(details_subframe)
            max_frame.pack(fill=tk.X, pady=5)
            ttk.Label(max_frame, text="Máximos encontrados:", 
                     font=("Arial", 12, "bold"), foreground="#33B5FF").pack(anchor=tk.W, padx=5)
            
            x = symbols('x')
            f = lambdify_safe((x,), expr)
            
            for point in maxima:
                try:
                    point_val = float(point)
                    y_val = safe_eval(f, point_val)
                    
                    point_frame = ttk.Frame(details_subframe)
                    point_frame.pack(fill=tk.X, pady=2, padx=20)
                    
                    point_text = f"x = {point_val:.4f}, f(x) = {y_val:.4f}"
                    ttk.Label(point_frame, text=point_text, 
                             font=("Arial", 10), foreground="#FFFFFF").pack(anchor=tk.W)
                    
                    ttk.Label(point_frame, text="La función alcanza un máximo local en este punto.", 
                             font=("Arial", 9, "italic"), foreground="#CCCCCC").pack(anchor=tk.W)
                except Exception as e:
                    print(f"Error al mostrar máximo: {e}")
        else:
            ttk.Label(details_subframe, text="No se encontraron máximos en el dominio analizado.", 
                     font=("Arial", 11), foreground="#FFFFFF").pack(anchor=tk.W, padx=5, pady=5)
        
        if minima:
            min_frame = ttk.Frame(details_subframe)
            min_frame.pack(fill=tk.X, pady=5)
            ttk.Label(min_frame, text="Mínimos encontrados:", 
                     font=("Arial", 12, "bold"), foreground="#33B5FF").pack(anchor=tk.W, padx=5)
            
            x = symbols('x')
            f = lambdify_safe((x,), expr)
            
            for point in minima:
                try:
                    point_val = float(point)
                    y_val = safe_eval(f, point_val)
                    
                    point_frame = ttk.Frame(details_subframe)
                    point_frame.pack(fill=tk.X, pady=2, padx=20)
                    
                    point_text = f"x = {point_val:.4f}, f(x) = {y_val:.4f}"
                    ttk.Label(point_frame, text=point_text, 
                             font=("Arial", 10), foreground="#FFFFFF").pack(anchor=tk.W)
                    
                    ttk.Label(point_frame, text="La función alcanza un mínimo local en este punto.", 
                             font=("Arial", 9, "italic"), foreground="#CCCCCC").pack(anchor=tk.W)
                except Exception as e:
                    print(f"Error al mostrar mínimo: {e}")
        else:
            ttk.Label(details_subframe, text="No se encontraron mínimos en el dominio analizado.", 
                     font=("Arial", 11), foreground="#FFFFFF").pack(anchor=tk.W, padx=5, pady=5)
        
        if inflection:
            inf_frame = ttk.Frame(details_subframe)
            inf_frame.pack(fill=tk.X, pady=5)
            ttk.Label(inf_frame, text="Posibles puntos de inflexión:", 
                     font=("Arial", 12, "bold"), foreground="#33B5FF").pack(anchor=tk.W, padx=5)
            
            x = symbols('x')
            f = lambdify_safe((x,), expr)
            
            for point in inflection:
                try:
                    point_val = float(point)
                    y_val = safe_eval(f, point_val)
                    
                    point_frame = ttk.Frame(details_subframe)
                    point_frame.pack(fill=tk.X, pady=2, padx=20)
                    
                    point_text = f"x = {point_val:.4f}, f(x) = {y_val:.4f}"
                    ttk.Label(point_frame, text=point_text, 
                             font=("Arial", 10), foreground="#FFFFFF").pack(anchor=tk.W)
                    
                    ttk.Label(point_frame, text="La segunda derivada es cero en este punto. Puede ser un punto de inflexión o requiere análisis adicional.", 
                             font=("Arial", 9, "italic"), foreground="#CCCCCC").pack(anchor=tk.W)
                except Exception as e:
                    print(f"Error al mostrar punto de inflexión: {e}")
        
        if undetermined:
            und_frame = ttk.Frame(details_subframe)
            und_frame.pack(fill=tk.X, pady=5)
            ttk.Label(und_frame, text="Puntos indeterminados:", 
                     font=("Arial", 12, "bold"), foreground="#33B5FF").pack(anchor=tk.W, padx=5)
            
            for point in undetermined:
                point_frame = ttk.Frame(details_subframe)
                point_frame.pack(fill=tk.X, pady=2, padx=20)
                
                ttk.Label(point_frame, text=f"x = {point}", 
                         font=("Arial", 10), foreground="#FFFFFF").pack(anchor=tk.W)
                
                ttk.Label(point_frame, text="No se pudo determinar la naturaleza de este punto crítico.", 
                         font=("Arial", 9, "italic"), foreground="#CCCCCC").pack(anchor=tk.W)
        
        explanation_frame = ttk.LabelFrame(popup_frame, text="Explicación", padding=10)
        explanation_frame.pack(fill=tk.X, pady=10)
        
        explanation_text = """
        El análisis de extremos examina los puntos críticos de la función, que son los puntos donde la primera derivada es igual a cero.
        
        - Un punto crítico es un máximo local si la segunda derivada es negativa en ese punto.
        - Un punto crítico es un mínimo local si la segunda derivada es positiva en ese punto.
        - Si la segunda derivada es cero, puede ser un punto de inflexión o requiere análisis adicional.
        
        El análisis muestra la ubicación de estos puntos y su valor de función correspondiente.
        """
        
        ttk.Label(explanation_frame, text=explanation_text, 
                 font=("Arial", 10), foreground="#CCCCCC", wraplength=650).pack(pady=5)
        
        ttk.Button(popup_frame, text="Cerrar", command=popup.destroy).pack(pady=10)
    
    def show_examples(self):
        popup = tk.Toplevel(self.root)
        popup.title("Ejemplos de Funciones")
        popup.geometry("650x600")
        popup.configure(bg="#121212")
        popup.grab_set()
        
        popup_frame = ttk.Frame(popup, padding=15)
        popup_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(popup_frame, text="Ejemplos de Funciones", 
                  font=("Arial", 14, "bold"), foreground="#33B5FF").pack(pady=10)
        
        ttk.Label(popup_frame, text="Seleccione una categoría y haga clic en un ejemplo para usarlo:", 
                  font=("Arial", 10), foreground="#FFFFFF").pack(pady=5)
        
        notebook = ttk.Notebook(popup_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        for category, examples_list in self.examples.items():
            category_frame = ttk.Frame(notebook)
            notebook.add(category_frame, text=category)
            
            canvas = tk.Canvas(category_frame, highlightthickness=0, bg="#1E1E1E")
            scrollbar = ttk.Scrollbar(category_frame, orient="vertical", command=canvas.yview)
            examples_scrollframe = ttk.Frame(canvas)
            
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            canvas.create_window((0, 0), window=examples_scrollframe, anchor="nw")
            examples_scrollframe.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
            
            for i, example in enumerate(examples_list):
                btn = ttk.Button(examples_scrollframe, text=example, width=30,
                                command=lambda ex=example: self.use_example(ex))
                btn.pack(pady=2, padx=5, anchor=tk.W)
        
        ttk.Button(popup_frame, text="Cerrar", command=popup.destroy).pack(pady=10)
    
    def use_example(self, example):
        self.expr_var.set(example)
    
    def add_to_history(self, expr):
        if expr not in self.history:
            self.history.append(expr)
            self.history_listbox.insert(tk.END, expr)
    
    def use_history_item(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            self.expr_var.set(self.history_listbox.get(index))
    
    def clear_history(self):
        self.history = []
        self.history_listbox.delete(0, tk.END)
    
    def export_results(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        if filepath:
            with open(filepath, 'w') as f:
                f.write(f"Función original: {self.original_expr.cget('text')}\n")
                f.write(f"Derivada: {self.derivative_expr.cget('text')}\n")
                f.write(f"Extremos: {self.extrema_expr.cget('text')}\n")
    
    def change_theme(self, theme):
        self.current_theme = theme
        if theme == 'dark':
            self.configure_dark_theme()
        elif theme == 'light':
            self.configure_light_theme()
        
        if hasattr(self, 'current_expr'):
            self.plot_functions(self.current_expr, getattr(self, 'current_derivative', None))
    
    def configure_light_theme(self):
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure('.', background='#F0F0F0', foreground='#000000')
        style.configure('TFrame', background='#F0F0F0')
        style.configure('TLabel', background='#F0F0F0', foreground='#000000')
        style.configure('TButton', background='#E0E0E0', foreground='#000000')
        style.configure('TEntry', fieldbackground='#FFFFFF', foreground='#000000')
        style.configure('TLabelframe', background='#F0F0F0', foreground='#0066CC')
        style.configure('TLabelframe.Label', background='#F0F0F0', foreground='#0066CC')
        style.configure('TNotebook', background='#F0F0F0')
        style.configure('TNotebook.Tab', background='#E0E0E0', foreground='#000000')
        style.map('TNotebook.Tab', background=[('selected', '#0066CC')], foreground=[('selected', '#FFFFFF')])
        
        self.graph_manager.current_theme = 'light'
        if self.is_3d_var.get():
            self.graph_manager.init_3d_plot()
        else:
            self.graph_manager.init_2d_plot()
    
    def show_help(self):
        help_text = """
        INSTRUCCIONES DE USO:
        
        1. Ingrese su función en el campo de texto.
        2. Seleccione el orden de derivación.
        3. Haga clic en 'Calcular Derivada'.
        
        CONSEJOS:
        - Use '*' para multiplicación explícita.
        - Use '^' para potenciación.
        - Vea ejemplos en el menú 'Ejemplos'.
        
        ATAJOS DE TECLADO:
        - Ctrl+N: Nuevo cálculo
        - Ctrl+D: Calcular derivada
        - Ctrl+E: Exportar resultados
        - F1: Mostrar ayuda
        """
        messagebox.showinfo("Ayuda", help_text)
    
    def show_about(self):
        messagebox.showinfo("Acerca de", "Calculadora de Derivadas Avanzada\nVersión 2.0\nDesarrollado por [PETRIZ]")
    
    def clear_input(self):
        self.expr_var.set("")
        self.order_var.set(1)
        self.x_min_var.set(-10)
        self.x_max_var.set(10)
        self.y_min_var.set(-10)
        self.y_max_var.set(10)
        
        self.original_expr.config(text="")
        self.derivative_expr.config(text="")
        self.extrema_expr.config(text="")
        
        self.graph_manager.ax.clear()
        self.graph_manager.ax.set_facecolor('#1E1E1E' if self.current_theme == 'dark' else '#F0F0F0')
        self.graph_manager.ax.axhline(y=0, color='#555555', linestyle='-', alpha=0.5)
        self.graph_manager.ax.axvline(x=0, color='#555555', linestyle='-', alpha=0.5)
        self.graph_manager.ax.grid(True, color='#555555', alpha=0.3)
        self.graph_manager.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DerivativeCalculator(root)
    root.mainloop()
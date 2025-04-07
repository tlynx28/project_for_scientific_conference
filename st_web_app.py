import streamlit as st
from sympy import (exp, log, sin, cos, tan, sqrt, sinh, cosh,
                   pi, Abs, cot, factorial, sec, csc, E, symbols,
                   latex, series, oo, limit, I, im, arg, re,
                   residue, sympify)
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application,
                                        convert_xor, split_symbols,
                                        function_exponentiation, TokenError)
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Ряды Тейлора и Лорана",
    page_icon="icon.png",
    layout="wide"
)

transformations = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor,
     split_symbols, function_exponentiation)
)


def fix_math_functions(text):
    text = re.sub(r'[а-яА-Я]', '', text).strip()
    text = text.replace('^', '**').replace('∨', '|').replace('∧', '&')
    text = re.sub(
        r'(\d+)(sin|cos|tan|tg|cot|ctg|exp|e|E|log|ln|sqrt|abs)',
        r'\1*\2', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(sin|cos|tan|tg|cot|ctg|exp|e|E|log|ln|sqrt|abs)\s*(\()',
        r'\1\2', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(sin|cos|tan|tg|cot|ctg|exp|e|E|log|ln|sqrt|abs)([a-zA-Z0-9]+)',
        r'\1(\2)', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)([a-zA-Z(])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z)])(\d+)', r'\1*\2', text)

    return text


def parse_function(func_str, local_dict=None):
    if not func_str.strip():
        return None, "Введите функцию, поле не может быть пустым"

    error_messages = {
        f"name '{current_var.name}' is not defined": f"Используйте '{current_var.name}' как переменную",
        "invalid syntax": "Ошибка в синтаксисе выражения",
        "unexpected EOF": "Незавершённое выражение",
        "could not convert string to float": "Недопустимые символы в выражении",
        "TokenError": "Недопустимый символ в выражении"
    }

    if local_dict is None:
        local_dict = {
            'x': x, 'z': z, 'exp': exp, 'log': log, 'ln': log,
            'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot,
            'sec': sec, 'csc': csc, 'sinh': sinh, 'cosh': cosh,
            'sqrt': sqrt, 'pi': pi, 'abs': Abs, '!': factorial,
            'tg': tan, 'ctg': cot, 'e': E, 'E': E, 'i': I,
            'Re': re, 're': re, 'Im': im, 'im': im, 'arg': arg
        }

    try:
        p_expr = parse_expr(func_str, transformations=transformations, local_dict=local_dict)

        undefined = [str(s) for s in p_expr.free_symbols if str(s) not in local_dict]
        if undefined:
            return None, f"Неопределённые символы: {', '.join(map(str, undefined))}"

        return p_expr, None

    except TokenError as e:
        return None, f"Ошибка в позиции {e.args[1]}: неправильный символ"
    except SyntaxError:
        return None, "Ошибка синтаксиса: проверьте скобки и операторы"
    except Exception as e:
        for pattern, msg in error_messages.items():
            if pattern in str(e):
                return None, msg
        return None, f"Ошибка: {str(e)}"


def data_output(txt, func_txt):
    st.markdown(txt)
    st.latex(func_txt)
    return


def is_taylor_valid(f, x_func, a, n):
    try:
        try:
            f.subs(x_func, a).evalf()
        except:
            return st.error("Нельзя разложить в ряд Тейлора: функция не определена в точке a")

        for j in range(n + 1):
            derivative = f.diff(x_func, j)
            try:
                derivative_at_a = derivative.subs(x_func, a).evalf()
                if not derivative_at_a.is_finite:
                    return st.error(f"Нельзя разложить в ряд Тейлора: {j}-я производная не существует или бесконечна")
            except:
                return st.error(f"Нельзя разложить в ряд Тейлора: не удалось вычислить {j}-ю производную в точке a")

        try:
            taylor = series(f, x_func, a, n + 1).removeO()
        except:
            return st.error("Нельзя разложить в ряд Тейлора: sympy не смог построить ряд")

        if taylor == 0:
            test_points = [a + 0.01, a - 0.01, a + 0.1, a - 0.1]
            for point in test_points:
                if f.subs(x_func, point).evalf() != 0:
                    return st.warning("Ряд Тейлора нулевой, но функция не нулевая в окрестности точки a")

        test_points = [a + 0.01, a - 0.01, a + 0.1, a - 0.1]
        for point in test_points:
            exact_val = f.subs(x_func, point).evalf(10)
            approx_val = taylor.subs(x_func, point).evalf(10)
            rel_error = Abs((exact_val - approx_val) / exact_val) if exact_val != 0 else Abs(exact_val - approx_val)
            if rel_error > 1e-5:
                return st.warning("Ряд Тейлора не сходится к функции в окрестности точки a")

        data_output("### Ряд Тейлора:", f"T({current_var.name}) = {latex(taylor)}")
        plot_taylor_and_function(f, taylor, x_func, a)
        return

    except TypeError:
        return st.error("Не удалось построить ряд Тейлора")


def plot_taylor_and_function(f, taylor, var, a):
    try:
        if var == x:
            x_vals = np.linspace(float(a) - 2, float(a) + 2, 400)
            y_func = np.array([float(f.subs(var, val).evalf()) for val in x_vals])
            y_taylor = np.array([float(taylor.subs(var, val).evalf()) for val in x_vals])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_func, mode='lines', name=f'Функция f({var.name})'))
            fig.add_trace(go.Scatter(x=x_vals, y=y_taylor, mode='lines', name=f'Ряд Тейлора'))

            fig.update_layout(
                title=f'Функция и её ряд Тейлора в точке {var.name}={a}',
                xaxis_title=var.name,
                yaxis_title='y',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            x_vals = np.linspace(float(a) - 2, float(a) + 2, 100)
            y_vals = np.linspace(-1, 1, 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = X + 1j * Y

            F = np.vectorize(lambda z_val: abs(complex(f.subs(var, z_val).evalf())))
            z_func = F(Z)

            T = np.vectorize(lambda z_val: abs(complex(taylor.subs(var, z_val).evalf())))
            z_taylor = T(Z)

            fig = make_subplots(rows=1, cols=2,
                                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                                subplot_titles=('Функция', 'Ряд Тейлора'))

            fig.add_trace(go.Surface(z=z_func, x=X, y=Y, name=f'|f({var.name})|'), row=1, col=1)
            fig.add_trace(go.Surface(z=z_taylor, x=X, y=Y, name=f'|Ряд Тейлора|'), row=1, col=2)

            fig.update_layout(
                title_text=f'Модуль функции и ряда Тейлора вблизи {var.name}={a}',
                scene1=dict(xaxis_title='Re(z)', yaxis_title='Im(z)', zaxis_title='|f(z)|'),
                scene2=dict(xaxis_title='Re(z)', yaxis_title='Im(z)', zaxis_title='|T(z)|')
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Не удалось построить график: {str(e)}")


def analyze_singularities(f, z_var, a):
    try:
        a_sym = sympify(a)

        try:
            lim = limit(f, z_var, a_sym)
            if lim.is_finite:
                try:
                    if f.subs(z_var, a_sym).is_finite and lim != f.subs(z_var, a_sym):
                        return "removable", 0, lim
                except:
                    pass
                return "removable", 0, lim
        except:
            pass

        try:
            m = 0
            while True:
                test_expr = (z_var - a_sym) ** (m + 1) * f
                try:
                    test_limit = limit(test_expr, z_var, a_sym)
                    if test_limit.is_finite and not test_limit.is_zero:
                        m += 1
                    else:
                        break
                except:
                    break

            if m > 0:
                res = residue(f, z_var, a_sym)
                if res.is_finite and res != 0:
                    return "pole", m, res
                else:
                    test_limit = limit((z_var - a_sym) ** m * f, z_var, a_sym)
                    if test_limit.is_finite and not test_limit.is_zero:
                        return "pole", m, test_limit
        except:
            pass

        try:
            test_expr = f.subs(z_var, a_sym + 1 / z_var)
            test_limit1 = limit(test_expr, z_var, 0, '+')
            test_limit2 = limit(test_expr, z_var, 0, '-')

            if test_limit1 is oo or test_limit2 is oo or (not test_limit1.is_finite and not test_limit2.is_finite):
                return "essential", None, None

            if test_limit1 != test_limit2:
                return "essential", None, None
        except:
            pass

        if f.has(log(z_var - a_sym)) or any(f.has(fn) for fn in [sqrt, arg]):
            return "branch", None, None
        try:
            if not f.is_analytic(z_var, a_sym) and not f.is_meromorphic(z_var, a_sym):
                return "branch", None, None
        except:
            pass

        return "unknown", None, None

    except Exception as e:
        print(f"Ошибка анализа: {str(e)}")
        return "error", None, None


def is_laurent_value(f, x_func, a, n):
    try:
        laurent = series(f, x_func, a, n=n + 1).removeO()

        sing_type, order, res = analyze_singularities(f, x_func, a)

        if sing_type == "pole":
            info = f"Полюс {order}-го порядка в {x_func.name}={a}"
            if res is not None:
                info += f" (вычет = {latex(res)})"
            return (data_output("### Ряд Лорана:", f"L({x_func.name}) = {latex(laurent)}"),
                    st.info(info),
                    plot_laurent_and_function(f, laurent, x_func, a))
        elif sing_type == "essential":
            return (data_output("### Ряд Лорана:", f"L({x_func.name}) = {latex(laurent)}"),
                    st.warning(f"Существенная особенность в {x_func.name}={a}"),
                    plot_laurent_and_function(f, laurent, x_func, a))
        elif sing_type == "branch":
            return (data_output("### Ряд Лорана:", f"L({x_func.name}) = {latex(laurent)}"),
                    st.warning(f"Точка ветвления в {x_func.name}={a}"),
                    plot_laurent_and_function(f, laurent, x_func, a))

        data_output("### Ряд Лорана:", f"L({x_func.name}) = {latex(laurent)}")
        plot_laurent_and_function(f, laurent, x_func, a)
        return

    except Exception as e:
        st.error(f"Ошибка при построении ряда: {str(e)}")
        return None


def plot_laurent_and_function(f, laurent, var, a):
    try:
        if var == x:
            x_vals = np.linspace(float(a) - 2, float(a) + 2, 400)
            y_func = np.array([float(f.subs(var, val).evalf()) for val in x_vals])
            y_laurent = np.array([float(laurent.subs(var, val).evalf()) for val in x_vals])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_func, mode='lines', name=f'Функция f({var.name})'))
            fig.add_trace(go.Scatter(x=x_vals, y=y_laurent, mode='lines', name=f'Ряд Лорана'))

            fig.update_layout(
                title=f'Функция и её ряд Лорана в точке {var.name}={a}',
                xaxis_title=var.name,
                yaxis_title='y',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            x_vals = np.linspace(float(a) - 2, float(a) + 2, 100)
            y_vals = np.linspace(-1, 1, 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = X + 1j * Y

            F = np.vectorize(lambda z_val: abs(complex(f.subs(var, z_val).evalf())))
            z_func = F(Z)

            L = np.vectorize(lambda z_val: abs(complex(laurent.subs(var, z_val).evalf())))
            z_laurent = L(Z)

            fig = make_subplots(rows=1, cols=2,
                                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                                subplot_titles=('Функция', 'Ряд Лорана'))

            fig.add_trace(go.Surface(z=z_func, x=X, y=Y, name=f'|f({var.name})|'), row=1, col=1)
            fig.add_trace(go.Surface(z=z_laurent, x=X, y=Y, name=f'|Ряд Лорана|'), row=1, col=2)

            fig.update_layout(
                title_text=f'Модуль функции и ряда Лорана вблизи {var.name}={a}',
                scene1=dict(xaxis_title='Re(z)', yaxis_title='Im(z)', zaxis_title='|f(z)|'),
                scene2=dict(xaxis_title='Re(z)', yaxis_title='Im(z)', zaxis_title='|L(z)|')
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Не удалось построить график: {str(e)}")


st.title("Разложение функции в ряды Тейлора и Лорана")

variable_type = st.radio(
    "Тип переменной:",
    ["Действительная (x)", "Комплексная (z)"],
    index=0,
    horizontal=True,
    key="var_type"
)

if "prev_var_type" not in st.session_state:
    st.session_state.prev_var_type = variable_type

if st.session_state.prev_var_type != variable_type:
    old_var = "x" if st.session_state.prev_var_type == "Действительная (x)" else "z"
    new_var = "x" if variable_type == "Действительная (x)" else "z"


    def replace_var_safety(e):
        return re.sub(rf'(?<!\w){old_var}(?!\w)', new_var, e)


    if hasattr(st.session_state, 'function'):
        st.session_state.function = replace_var_safety(st.session_state.function)
    if hasattr(st.session_state, 'last_valid_function'):
        st.session_state.last_valid_function = replace_var_safety(st.session_state.last_valid_function)

    st.session_state.prev_var_type = variable_type
    st.rerun()

x = symbols('x', real=True)
z = symbols('z', complex=True)
current_var = x if variable_type == "Действительная (x)" else z

if "function" not in st.session_state:
    st.session_state.function = f"sin({current_var})"
if "last_valid_function" not in st.session_state:
    st.session_state.last_valid_function = f"sin({current_var})"

st.markdown("### Текущая функция:")
expr, error = parse_function(st.session_state.last_valid_function)
if error:
    st.error(error)
else:
    st.latex(f'f({current_var.name}) = {latex(expr)}')

new_function = st.text_input(
    "Введите функцию:",
    value=st.session_state.function,
    help=f"Используйте '{current_var.name}' как переменную. Не смешивайте x и z!"
)

if new_function != st.session_state.function:
    fixed_function = fix_math_functions(new_function)
    st.session_state.function = new_function

    expr, error = parse_function(fixed_function)
    if error:
        st.error(error)
        st.warning("Во избежании проблем рекомендуется перезагрузить страницу")
    else:
        st.session_state.last_valid_function = fixed_function
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    x0 = st.number_input(f"Точка разложения ({current_var.name}₀):", value=0.0)
with col2:
    n_terms = st.number_input("Количество членов ряда (n):", min_value=1, max_value=20, value=5)

with st.expander("📚 Примеры функций", expanded=False):
    examples = [
        f"{current_var.name}", f"{current_var.name}**2",
        f"{current_var.name}**3 - 2*{current_var.name}",
        f"1 + {current_var.name} + {current_var.name}**2/2 + {current_var.name}**3/6",
        f"1/(1-{current_var.name})", f"1/(1+{current_var.name}**2)",
        f"({current_var.name}+2)/({current_var.name}-3)",
        f"1/({current_var.name}**2 - 1)",
        f"sin({current_var.name})", f"cos(2*{current_var.name})",
        f"tan({current_var.name})",
        f"sin({current_var.name})/{current_var.name}",
        f"sin({current_var.name})*cos({current_var.name})",
        f"exp({current_var.name})", f"exp(-{current_var.name}**2)",
        f"log(1+{current_var.name})",
        f"exp({current_var.name})*sin({current_var.name})",
        f"sinh({current_var.name})", f"cosh({current_var.name})**2",
        f"sqrt(1 + {current_var.name})", f"abs({current_var.name})",
        f"1/({current_var.name}+1) - 1/({current_var.name}-1)",
        f"log(1 + tan(pi*{current_var.name}/3))",
        f"exp(1/{current_var.name})", f"exp(sin({current_var.name}))"
    ]
    cols = st.columns(5)
    for i, example in enumerate(examples):
        with cols[i % 5]:
            if st.button(example):
                st.session_state.function = example
                st.session_state.last_valid_function = example
                st.rerun()

if st.button("Разложить функцию в ряды Тейлора и Лорана"):
    st.markdown("### Параметры ввода:")
    data_output("Функция:",f"f({current_var.name}) = {latex(expr)}")
    data_output("Точка разложения:",f"{current_var.name}_0 = {x0}")
    data_output("Количество членов ряда:", f"n = {n_terms}")
    is_taylor_valid(expr, current_var, x0, n_terms)
    is_laurent_value(expr, current_var, x0, n_terms)

st.markdown("---")
st.caption("Проект для научной конференции | © 2025")

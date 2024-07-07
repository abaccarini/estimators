import numpy as np
from gekko import GEKKO
from sympy import Symbol, Function, Number, eye, Matrix, factor

m = GEKKO()  # Initialize gekko
m.options.SOLVER = 1  # APOPT is an MINLP solver

# optional solver settings with APOPT
m.solver_options = [
    "minlp_maximum_iterations 10000",  # minlp iterations with integer solution
    "minlp_max_iter_with_int_sol 500",  # treat minlp as nlp
    "minlp_as_nlp 0",  # nlp sub-problem max iterations
    "nlp_maximum_iterations 50",  # 1 = depth first, 2 = breadth first
    "minlp_branch_method 2",  # maximum deviation from whole number
    "minlp_integer_tol 0.00005",  # covergence tolerance
    "minlp_gap_tol 0.00001",
]

# Initialize variables

Delta = 24.0
t = 1

target_configs = [
    (1, 1, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]
def numerator_integer_non_gekko(s, tau_tuple):
    # s[0] = s1 , s[0] = s12 , s[3] = s123
    # s[1] = s2 , s[1] = s13
    # s[2] = s3 , s[2] = s23
    num = np.array(
        [
            [
                tau_tuple[0] * t + Delta,
                tau_tuple[0] * tau_tuple[1] * t + s[0] + s[3],
                tau_tuple[0] * tau_tuple[2] * t + s[1] + s[3],
            ],
            [
                tau_tuple[0] * tau_tuple[1] * t + s[0] + s[3],
                tau_tuple[1] * t + Delta,
                tau_tuple[1] * tau_tuple[2] * t + s[2] + s[3],
            ],
            [
                tau_tuple[0] * tau_tuple[2] * t + s[1] + s[3],
                tau_tuple[1] * tau_tuple[2] * t + s[2] + s[3],
                tau_tuple[2] * t +Delta,
            ],
        ]
    )
    return num


def f_integer_non_gekko(s):
    # s[0] = s1 , s[0] = s12 , s[3] = s123
    # s[1] = s2 , s[1] = s13
    # s[2] = s3 , s[2] = s23
    denom = np.array(
        [
            [
                Delta,
                s[0] + s[3],
                s[1] + s[3],
            ],
            [
                s[0] + s[3],
                Delta,
                s[2] + s[3],
            ],
            [
                s[1] + s[3],
                s[2] + s[3],
               Delta,
            ],
        ]
    )
    values = []
    for t_conf in target_configs:
        values.append(np.linalg.det(numerator_integer_non_gekko(s, t_conf)))
    vmax = max(values) / np.linalg.det(denom)
    return (1.0) * 0.5 * np.log2(vmax)
    # return (-1.0) * vmax



def numerator_integer(s, tau_tuple):
    # s[0] = s1 , s[0] = s12 , s[3] = s123
    # s[1] = s2 , s[1] = s13
    # s[2] = s3 , s[2] = s23
    num = np.array(
        [
            [
                tau_tuple[0] * t + Delta,
                tau_tuple[0] * tau_tuple[1] * t + s[0].value + s[3].value,
                tau_tuple[0] * tau_tuple[2] * t + s[1].value + s[3].value,
            ],
            [
                tau_tuple[0] * tau_tuple[1] * t + s[0].value + s[3].value,
                tau_tuple[1] * t + Delta,
                tau_tuple[1] * tau_tuple[2] * t + s[2].value + s[3].value,
            ],
            [
                tau_tuple[0] * tau_tuple[2] * t + s[1].value + s[3].value,
                tau_tuple[1] * tau_tuple[2] * t + s[2].value + s[3].value,
                tau_tuple[2] * t +Delta,
            ],
        ]
    )
    return num


def f_integer(s):
    # s[0] = s1 , s[0] = s12 , s[3] = s123
    # s[1] = s2 , s[1] = s13
    # s[2] = s3 , s[2] = s23
    denom = np.array(
        [
            [
                Delta,
                s[0].value + s[3].value,
                s[1].value + s[3].value,
            ],
            [
                s[0].value + s[3].value,
                Delta,
                s[2].value + s[3].value,
            ],
            [
                s[1].value + s[3].value,
                s[2].value + s[3].value,
               Delta,
            ],
        ]
    )
    values = []
    for t_conf in target_configs:
        values.append(np.linalg.det(numerator_integer(s, t_conf)))
    vmax = max(values) / (np.linalg.det(denom))
    return (1.0) * 0.5 * np.log2(vmax)
    # return (-1.0) * vmax


def f_integer_no_taus(s):
    # s[0] = s12 , s[3] = s123  , s[6] = tau_3
    # s[1] = s13 , s[4] = tau_1 ,
    # s[2] = s23 , s[5] = tau_2 ,
    denom = np.array(
        [
            [
                Delta,
                s[0].value + s[3].value,
                s[1].value + s[3].value,
            ],
            [
                s[0].value + s[3].value,
                Delta,
                s[2].value + s[3].value,
            ],
            [
                s[1].value + s[3].value,
                s[2].value + s[3].value,
                Delta,
            ],
        ]
    )
    num = np.array(
        [
            [
                s[4].value * t + Delta,
                s[4].value * s[5].value * t + s[0].value + s[3].value,
                s[4].value * s[6].value * t + s[1].value + s[3].value,
            ],
            [
                s[4].value * s[5].value * t + s[0].value + s[3].value,
                s[5].value * t + Delta,
                s[5].value * s[6].value * t + s[2].value + s[3].value,
            ],
            [
                s[4].value * s[6].value * t + s[1].value + s[3].value,
                s[5].value * s[6].value * t + s[2].value + s[3].value,
                s[6].value * t + Delta,
            ],
        ]
    )

    val = np.linalg.det(num) / np.linalg.det(denom)
    return (1.0) * 0.5 * np.log2(val)
    # return (-1.0) * vmax


# x0_taus = m.Array(m.Var, (7), integer=True)
x0_taus = m.Array(m.Var, (4), integer=True)
# x0_taus = 10*[None]
print(x0_taus)


x0_taus[0].lower = 0
x0_taus[0].upper = Delta

x0_taus[1].lower = 0
# x0_taus[1].upper = Delta - x0_taus[0]
x0_taus[1].upper = Delta

x0_taus[2].lower = 0
# x0_taus[2].upper = Delta - x0_taus[0] - x0_taus[1]
x0_taus[2].upper = Delta

x0_taus[3].lower = 0
# x0_taus[3].upper = Delta - x0_taus[0] - x0_taus[1] - x0_taus[2]
x0_taus[3].upper = Delta

x0_taus[0].value = 1
x0_taus[1].value = 1
x0_taus[2].value = 1
x0_taus[3].value = 8

 
# m.Equation(x0_taus[0] - Delta >= 0)
# m.Equation(x0_taus[1] - (Delta - x0_taus[0]) >= 0)
# m.Equation(x0_taus[2] - (Delta - x0_taus[0] - x0_taus[1]) >= 0)

m.Equation(x0_taus[3] - (Delta - x0_taus[0] - x0_taus[1] - x0_taus[2]) >= 0)
m.Equation(x0_taus[2] - (Delta - x0_taus[3] - x0_taus[1] - x0_taus[0]) >= 0)
m.Equation(x0_taus[1] - (Delta - x0_taus[0] - x0_taus[3] - x0_taus[2]) >= 0)
m.Equation(x0_taus[0] - (Delta - x0_taus[3] - x0_taus[1] - x0_taus[2]) >= 0)


# Equations
# m.Equation(x0_taus[4] + x0_taus[5] + x0_taus[6] >= 1)
# m.Equation(x0_taus[4] + x0_taus[5] + x0_taus[6] <= 3)
# # m.Equation(x1**2+x2**2+x3**2+x4**2==40)

# m.Obj(f_integer_no_taus(x0_taus))  # Objective
m.Minimize(f_integer(x0_taus))  # Objective
m.solve(disp=True)  # Solve


print("Results")
print("[0]: " + str(x0_taus[0].value))
print("[1]: " + str(x0_taus[1].value))
print("[2]: " + str(x0_taus[2].value))
print("[3]: " + str(x0_taus[3].value))
# print("[4]: " + str(x0_taus[4].value))
# print("[5]: " + str(x0_taus[5].value))
# print("[6]: " + str(x0_taus[6].value))
print("Objective: " + str(m.options.objfcnval))

x_0_true = np.array([0.0, 0.0, 0.0, 8.0])
print("expected: x_0 = ", x_0_true, " -- f(x) =", f_integer_non_gekko(x_0_true))
print("error = ", abs(f_integer_non_gekko(x_0_true) - m.options.objfcnval) )
import numpy as np
import pulp

# from gekko import GEKKO
from scipy.optimize import minimize, Bounds

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


def h_X_T(sigma):
    return np.log(np.sqrt(sigma * 2.0 * np.pi * np.e)) / np.log(2.0)


def numerator(alpha, tau_tuple):
    num = np.array(
        [
            [
                tau_tuple[0] * t / Delta + 1,
                tau_tuple[0] * tau_tuple[1] * t / Delta + alpha[0] + alpha[3],
                tau_tuple[0] * tau_tuple[2] * t / Delta + alpha[1] + alpha[3],
            ],
            [
                tau_tuple[0] * tau_tuple[1] * t / Delta + alpha[0] + alpha[3],
                tau_tuple[2] * t / Delta + 1,
                tau_tuple[1] * tau_tuple[2] * t / Delta + alpha[2] + alpha[3],
            ],
            [
                tau_tuple[0] * tau_tuple[2] * t / Delta + alpha[1] + alpha[3],
                tau_tuple[1] * tau_tuple[2] * t / Delta + alpha[2] + alpha[3],
                tau_tuple[2] * t / Delta + 1,
            ],
        ]
    )

    return num


def f(alpha):
    denom = np.array(
        [
            [
                1,
                alpha[0] + alpha[3],
                alpha[1] + alpha[3],
            ],
            [
                alpha[0] + alpha[3],
                1,
                alpha[2] + alpha[3],
            ],
            [
                alpha[1] + alpha[3],
                alpha[2] + alpha[3],
                1,
            ],
        ]
    )
    values = []
    for t_conf in target_configs:
        values.append(np.linalg.det(numerator(alpha, t_conf)))
    vmax = max(values) / np.linalg.det(denom)
    # return 0.5*np.log2(vmax)
    return (-1.0) * vmax


def numerator_integer(s, tau_tuple):
    # s[0] = s1 , s[3] = s12 , s[6] = s123
    # s[1] = s2 , s[4] = s13
    # s[2] = s3 , s[5] = s23
    num = np.array(
        [
            [
                tau_tuple[0] * t + s[0] + s[3] + s[4] + s[6],
                tau_tuple[0] * tau_tuple[1] * t + s[3] + s[6],
                tau_tuple[0] * tau_tuple[2] * t + s[4] + s[6],
            ],
            [
                tau_tuple[0] * tau_tuple[1] * t + s[3] + s[6],
                tau_tuple[1] * t + s[1] + s[3] + s[5] + s[6],
                tau_tuple[1] * tau_tuple[2] * t + s[5] + s[6],
            ],
            [
                tau_tuple[0] * tau_tuple[2] * t + s[4] + s[6],
                tau_tuple[1] * tau_tuple[2] * t + s[5] + s[6],
                tau_tuple[2] * t + s[2] + s[4] + s[5] + s[6],
            ],
        ]
    )
    return num


def f_integer(s):
    # s[0] = s1 , s[3] = s12 , s[6] = s123
    # s[1] = s2 , s[4] = s13
    # s[2] = s3 , s[5] = s23
    denom = np.array(
        [
            [
                s[0] + s[3] + s[4] + s[6],
                s[3] + s[6],
                s[4] + s[6],
            ],
            [
                s[3] + s[6],
                s[1] + s[3] + s[5] + s[6],
                s[5] + s[6],
            ],
            [
                s[4] + s[6],
                s[5] + s[6],
                s[2] + s[4] + s[5] + s[6],
            ],
        ]
    )
    values = []
    for t_conf in target_configs:
        values.append(np.linalg.det(numerator_integer(s, t_conf)))
    vmax = max(values) / np.linalg.det(denom)
    return (1.0) * 0.5 * np.log2(vmax)
    # return (-1.0) * vmax


def f_integer_no_taus(s):
    # s[0] = s1 , s[3] = s12 , s[6] = s123  , s[9] = tau_3
    # s[1] = s2 , s[4] = s13 , s[7] = tau_1 ,
    # s[2] = s3 , s[5] = s23 , s[8] = tau_2 ,
    denom = np.array(
        [
            [
                s[0] + s[3] + s[4] + s[6],
                s[3] + s[6],
                s[4] + s[6],
            ],
            [
                s[3] + s[6],
                s[1] + s[3] + s[5] + s[6],
                s[5] + s[6],
            ],
            [
                s[4] + s[6],
                s[5] + s[6],
                s[2] + s[4] + s[5] + s[6],
            ],
        ]
    )
    num = np.array(
        [
            [
                s[7] * t + s[0] + s[3] + s[4] + s[6],
                s[7] * s[8] * t + s[3] + s[6],
                s[7] * s[9] * t + s[4] + s[6],
            ],
            [
                s[7] * s[8] * t + s[3] + s[6],
                s[8] * t + s[1] + s[3] + s[5] + s[6],
                s[8] * s[9] * t + s[5] + s[6],
            ],
            [
                s[7] * s[9] * t + s[4] + s[6],
                s[8] * s[9] * t + s[5] + s[6],
                s[9] * t + s[2] + s[4] + s[5] + s[6],
            ],
        ]
    )

    val = np.linalg.det(num) / np.linalg.det(denom)
    return (1.0) * 0.5 * np.log2(val)
    # return (-1.0) * vmax


x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

x0_taus = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])


def ineq_constraint(x):
    """constrain all elements of x to be >= 0"""
    return x


def eq_constraint(x):
    """constrain the sum of all rows to be equal to 1"""
    return np.sum(x, 1) - 1


cons = (
    {"type": "ineq", "fun": lambda s: s[3]},
    {"type": "ineq", "fun": lambda s: (Delta) - s[3]},
    {"type": "ineq", "fun": lambda s: s[4]},
    {"type": "ineq", "fun": lambda s: (Delta - s[3]) - s[4]},
    {"type": "ineq", "fun": lambda s: s[5]},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[4]) - s[5]},
    {"type": "ineq", "fun": lambda s: s[6]},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[4] - s[5]) - s[6]},
    {"type": "ineq", "fun": lambda s: s[0] - (Delta - s[3] - s[4] - s[6])},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[4] - s[6]) - s[0]},
    {"type": "ineq", "fun": lambda s: s[1] - (Delta - s[3] - s[5] - s[6])},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[5] - s[6]) - s[1]},
    {"type": "ineq", "fun": lambda s: s[2] - (Delta - s[4] - s[5] - s[6])},
    {"type": "ineq", "fun": lambda s: (Delta - s[4] - s[5] - s[6]) - s[2]},
)

cons2 = (
    {"type": "ineq", "fun": lambda s: s[3]},
    {"type": "ineq", "fun": lambda s: (Delta) - s[3]},
    {"type": "ineq", "fun": lambda s: s[4]},
    {"type": "ineq", "fun": lambda s: (Delta - s[3]) - s[4]},
    {"type": "ineq", "fun": lambda s: s[5]},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[4]) - s[5]},
    {"type": "ineq", "fun": lambda s: s[6]},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[4] - s[5]) - s[6]},
    
    {"type": "ineq", "fun": lambda s: s[0] - (Delta - s[3] - s[4] - s[6])},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[4] - s[6]) - s[0]},
    {"type": "ineq", "fun": lambda s: s[1] - (Delta - s[3] - s[5] - s[6])},
    {"type": "ineq", "fun": lambda s: (Delta - s[3] - s[5] - s[6]) - s[1]},
    {"type": "ineq", "fun": lambda s: s[2] - (Delta - s[4] - s[5] - s[6])},
    {"type": "ineq", "fun": lambda s: (Delta - s[4] - s[5] - s[6]) - s[2]},
    
    {"type": "ineq", "fun": lambda s: s[7]}, # all tau values are between 0 and 1
    {"type": "ineq", "fun": lambda s: 1 - s[7]},
    {"type": "ineq", "fun": lambda s: s[8]},
    {"type": "ineq", "fun": lambda s: 1 - s[8]},
    {"type": "ineq", "fun": lambda s: s[9]},
    {"type": "ineq", "fun": lambda s: 1 - s[9]},
    
    {"type": "ineq", "fun": lambda s: s[7] + s[8] +s[9] - 1}, # the sum of the tau values must be AT LEAST 1
    {"type": "ineq", "fun": lambda s: 3 - (s[7] + s[8] +s[9])}, # the sum of the tau values must be AT MOST 3
    
)


# res_trust = minimize(
#     f_integer,
#     x0,
#     method="trust-constr",
#     constraints=cons,
# )

# res_cobyla = minimize(
#     f_integer,
#     x0,
#     method="COBYLA",
#     constraints=cons,
# )


# res_slsqp = minimize(
#     f_integer,
#     x0,
#     method="SLSQP",
#     constraints=cons,
# )


# # print(res)
# # # print(res.x)
# x_trust = res_trust.x
# print("trust :   x =", x_trust, " -- f(x) =", f_integer(x_trust))
# x_cobyla = res_cobyla.x
# print("cobyla:   x =", x_cobyla, " -- f(x) =", f_integer(x_cobyla))
# x_slsqp = res_slsqp.x
# print("slsqp:    x =", x_slsqp, " -- f(x) =", f_integer(x_slsqp))
# print("------------------------")
# x_0_true = np.array([16.0, 16.0, 16.0, 0.0, 0.0, 0.0, 8.0])
# print("expected: x_0 = ", x_0_true, " -- f(x) =", f_integer(x_0_true))



# res_trust = minimize(
#     f_integer_no_taus,
#     x0_taus,
#     method="trust-constr",
#     constraints=cons,
# )

res_cobyla = minimize(
    f_integer_no_taus,
    x0_taus,
    method="COBYLA",
    constraints=cons,
)


res_slsqp = minimize(
    f_integer_no_taus,
    x0_taus,
    method="SLSQP",
    constraints=cons,
)



# x_trust = res_trust.x
# print("trust :   x =", x_trust, " -- f(x) =", f_integer_no_taus(x_trust))
x_cobyla = res_cobyla.x
print("cobyla:   x =", x_cobyla, " -- f(x) =", f_integer_no_taus(x_cobyla))
x_slsqp = res_slsqp.x
print("slsqp:    x =", x_slsqp, " -- f(x) =", f_integer_no_taus(x_slsqp))
print("------------------------")
x_0_true = np.array([16.0, 16.0, 16.0, 0.0, 0.0, 0.0, 8.0, 1.0, 1.0, 1.0])
print("expected: x_0 = ", x_0_true, " -- f(x) =", f_integer_no_taus(x_0_true))
import numpy as np
from scipy.optimize import minimize

Delta = 24
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


def myfunc(alpha, tau1, tau2, tau3):
    num = np.array(
        [
            [
                tau1 * t / Delta + 1,
                tau1 * tau2 * t / Delta + alpha[0] + alpha[3],
                tau1 * tau3 * t / Delta + alpha[1] + alpha[3],
            ],
            [
                tau1 * tau2 * t / Delta + alpha[0] + alpha[3],
                tau3 * t / Delta + 1,
                tau2 * tau3 * t / Delta + alpha[2] + alpha[3],
            ],
            [
                tau1 * tau3 * t / Delta + alpha[1] + alpha[3],
                tau2 * tau3 * t / Delta + alpha[2] + alpha[3],
                tau3 * t / Delta + 1,
            ],
        ]
    )
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

    # multiplying by (-1) to turn minimize --] maximize
    return (-1.0) * 0.5 * np.log2(np.linalg.det(num) / np.linalg.det(denom))
    # return 0.5*np.log2(np.linalg.det(num) / np.linalg.det(denom))


def myfunc_v2(alpha):
    num = np.array(
        [
            [
                alpha[4] * t / Delta + 1,
                alpha[4] * alpha[5] * t / Delta + alpha[0] + alpha[3],
                alpha[4] * alpha[6] * t / Delta + alpha[1] + alpha[3],
            ],
            [
                alpha[4] * alpha[5] * t / Delta + alpha[0] + alpha[3],
                alpha[6] * t / Delta + 1,
                alpha[5] * alpha[6] * t / Delta + alpha[2] + alpha[3],
            ],
            [
                alpha[4] * alpha[6] * t / Delta + alpha[1] + alpha[3],
                alpha[5] * alpha[6] * t / Delta + alpha[2] + alpha[3],
                alpha[6] * t / Delta + 1,
            ],
        ]
    )
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

    # multiplying by (-1) to turn minimize --] maximize
    # return (-1.0)*0.5*np.log2(np.linalg.det(num) / np.linalg.det(denom))
    return 0.5 * np.log2(np.linalg.det(num) / np.linalg.det(denom))


def myfunc_v3(alpha, tau1, tau2, tau3):
    num = np.array(
        [
            [
                tau1 * t + Delta,
                tau1 * tau2 * t + alpha[0] + alpha[3],
                tau1 * tau3 * t + alpha[1] + alpha[3],
            ],
            [
                tau1 * tau2 * t + alpha[0] + alpha[3],
                tau3 * t + Delta,
                tau2 * tau3 * t + alpha[2] + alpha[3],
            ],
            [
                tau1 * tau3 * t + alpha[1] + alpha[3],
                tau2 * tau3 * t + alpha[2] + alpha[3],
                tau3 * t + Delta,
            ],
        ]
    )
    denom = np.array(
        [
            [
                Delta,
                alpha[0] + alpha[3],
                alpha[1] + alpha[3],
            ],
            [
                alpha[0] + alpha[3],
                Delta,
                alpha[2] + alpha[3],
            ],
            [
                alpha[1] + alpha[3],
                alpha[2] + alpha[3],
                Delta,
            ],
        ]
    )

    # multiplying by (-1) to turn minimize --] maximize
    return (-1.0) * 0.5 * np.log2(np.linalg.det(num) / np.linalg.det(denom))
    # return 0.5*np.log2(np.linalg.det(num) / np.linalg.det(denom))


# x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
# x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
# bnds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

# x0 = np.array([0.1, 0.1, 0.1, 0.1])
# x0 = np.array([1.0, 1.0, 1.0, 1.0])
x0 = np.array([0.0, 0.0, 0.0, 0.0])
bnds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


res = minimize(
    myfunc,
    x0,
    # method="TNC",
    method="nelder-mead",
    # method="BFGS",
    # method="Newton-CG",
    bounds=bnds,
    args=target_configs[0],
    options={"xatol": 1e-8, "disp": True},
)


print(res)
print(res.x)
x = res.x
print(myfunc(x, target_configs[0][0], target_configs[0][1], target_configs[0][2]))

# cons = (
#     {"type": "eq", "fun": lambda x: x[0] - 2 * x[1] + 2},
#     {"type": "eq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
#     {"type": "eq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
# )


# x0_2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# bnds_2 = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

# res = minimize(
#         myfunc_v2,
#         x0_2,
#         # method="TNC",
#         method="nelder-mead",
#         # method="BFGS",
#         # method="Newton-CG",
#         bounds=bnds_2,
#         options={"xatol": 1e-8, "disp": True},
#     )


# print(res)
# print(res.x)

import sympy as sp

sp.var("tau_1 tau_2 tau_3 t s_1 s_2 s_3 s_12 s_13 s_23 s_123")
O_mat = sp.Matrix(
    [
        [tau_1 * t + s_1 + s_12 + s_13 + s_123, tau_1 * tau_2 * t + s_12 + s_123, tau_1 * tau_3 * t + s_13 + s_123],
        [tau_1 * tau_2 * t + s_12 + s_123, tau_2 * t + s_2 + s_12 + s_23 + s_123, tau_2 * tau_3 * t + s_23 + s_123],
        [tau_1 * tau_3 * t + s_13 + s_123, tau_2 * tau_3 * t + s_23 + s_123, tau_3 * t + s_3 + s_13 + s_23 + s_123],
    ]
)

S_mat = sp.Matrix(
    [
        [ s_1 + s_12 + s_13 + s_123,  s_12 + s_123,  s_13 + s_123],
        [ s_12 + s_123,  s_2 + s_12 + s_23 + s_123,  s_23 + s_123],
        [ s_13 + s_123,  s_23 + s_123, s_3 + s_13 + s_23 + s_123],
    ]
)
deter1 = (O_mat.det())
deter2 = (S_mat.det())
print(deter1)
print()
print(deter2)
print()
final = deter2 - deter1
print(sp.collect(final, t))
# print(sp.collect(deter, tau_1))

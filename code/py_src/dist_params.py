class uniform_int_params:
    def __init__(self, a, b):
        self.t = "uniform_int"
        self.a = a
        self.b = b

    def getJSON(self):
        tmp = "(" + str(self.a) + "," + str(self.b) + ")"

        return {"dist_name": self.t, "a": self.a, "b": self.b, "param_str": tmp}


class poisson_params:
    def __init__(self, lam):
        self.t = "poisson"
        self.lam = lam

    def getJSON(self):
        return {
            "dist_name": self.t,
            "lam": self.lam,
            "param_str": "(" + str(self.lam) + ")",
        }


class uniform_real_params:
    def __init__(self, a, b):
        self.t = "uniform_real"
        self.a = a
        self.b = b

    def getJSON(self):
        return {
            "dist_name": self.t,
            "a": self.a,
            "b": self.b,
            "param_str": "(" + str(self.a) + "," + str(self.a) + ")",
        }


class lognormal_params:
    def __init__(self, mu, sigma):
        self.t = "lognormal"
        self.mu = mu
        self.sigma = sigma

    def getJSON(self):
        return {
            "dist_name": self.t,
            "m": self.mu,
            "s": self.sigma,
            "param_str": "(" + str(self.mu) + "," + str(self.sigma) + ")",
        }


class normal_params:
    def __init__(self, mu, sigma):
        self.t = "normal"
        self.mu = mu
        self.sigma = sigma

    def getJSON(self):
        return {
            "dist_name": self.t,
            "mu": self.mu,
            "sigma": self.sigma,
            "param_str": "(" + str(self.mu) + "," + str(self.sigma) + ")",
        }

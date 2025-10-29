class RefPhi:
    def __init__(self, phi, pseudo_min):
        self.phi = phi
        self.pseudo_min = pseudo_min


class RefTumor():
    def __init__(self, psi_mal, psi_env, key, pseudo_min):
        self.psi_mal = psi_mal
        self.psi_env = psi_env
        self.key = key
        self.pseudo_min = pseudo_min
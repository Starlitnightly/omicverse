import numpy as np
import pandas as pd

class ThetaPost:
    def __init__(self, theta, theta_cv):
        self.theta = theta
        self.theta_cv = theta_cv

    def new(bulk_id, cell_type, gibbs_list):
        n = len(bulk_id)
        k = len(cell_type)
        assert len(gibbs_list) == n

        theta = np.zeros((n, k))
        theta_cv = np.zeros((n, k))

        for i in range(n):
            theta[i] = gibbs_list[i]['theta_n']
            theta_cv[i] = gibbs_list[i]['theta.cv_n']

        theta = pd.DataFrame(theta, index = bulk_id, columns = cell_type)
        theta_cv = pd.DataFrame(theta_cv, index = bulk_id, columns = cell_type)

        return ThetaPost(theta, theta_cv)
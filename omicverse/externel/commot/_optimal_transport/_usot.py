import numpy as np


def uot(mu, nu, c, epsilon,
         niter=50, tau=-0.5, verb = 1, rho = np.Inf, stopThr= 1E-7):

    lmbda = rho / ( rho + epsilon )
    if np.isinf(rho): lmbda = 1

    mu = np.asarray(mu, float).reshape(-1,1)
    nu = np.asarray(nu, float).reshape(-1,1)
    N = [mu.shape[0], nu.shape[0]]
    H1 = np.ones([N[0],1]); H2 = np.ones([N[1],1])

    errs = []; Wprimal = []; Wdual = []
    u = np.zeros([N[0],1], float)
    v = np.zeros([N[1],1], float)
    for i in range(niter):
        u1 = u
        u = ave(tau, u, \
            lmbda * epsilon * np.log(mu) \
            - lmbda * epsilon * lse( M(u,v,H1,H2,c,epsilon) ) \
            + lmbda * u )
        v = ave(tau, v, \
            lmbda * epsilon * np.log(nu) \
            - lmbda * epsilon * lse( M(u,v,H1,H2,c,epsilon).T ) \
            + lmbda * v )
        gamma = np.exp( M(u,v,H1,H2,c,epsilon) )

        if np.isinf(rho):
            Wprimal.append(np.sum(c * gamma) - epsilon*H(gamma) )
            Wdual.append(np.sum(u*mu) + np.sum(v*nu) - epsilon*np.sum(gamma) )
            err = np.linalg.norm( np.sum(gamma,axis=1) - mu )
            errs.append( err )
        else:
            Wprimal.append(np.sum(c*gamma) - epsilon*H(gamma) \
                           + rho*KL(np.sum(gamma,axis=1), mu) \
                           + rho*KL(np.sum(gamma,axis=0), nu) )
            Wdual.append(- rho*KLd(u/rho,mu) - rho*KLd(v/rho,nu) \
                         - epsilon*np.sum(gamma) )
            err = np.linalg.norm(u-u1,1)
            errs.append( err )
        if err < stopThr and i > niter:
            break

    return gamma

def usot(mu, nu, c, c1, c2, alpha, epsilon = 0.1,
         niter = 10, gw_loss = 'square', rho = np.Inf):
    import ot
    mu = np.asarray(mu, float).reshape(-1,1)
    nu = np.asarray(nu, float).reshape(-1,1)
    gamma0 = np.outer(mu, nu)
    gamma_old = np.array(gamma0, float)
    G = np.empty(c.shape, float)
    for i in range(niter):
        # Construct loss
        G_w = ( 1.0 - alpha ) * c
        if gw_loss == 'square':
            fc1 = 0.5*c1**2; fc2 = 0.5*c2**2
            hc1 = c1; hc2 = c2
        constC1 = np.dot(np.dot(fc1, mu), np.ones(len(nu), float).reshape(1,-1))
        constC2 = np.dot(np.ones(len(mu)).reshape(-1,1), np.dot(nu.reshape(1,-1),fc2.T))
        constC = constC1 + constC2
        G_gw = alpha * 2.0 * (constC - np.dot(hc1, gamma_old).dot(hc2.T))
        G[:,:] = G_w[:,:] + G_gw[:,:]
        if np.isinf(rho):
            gamma_tuta = ot.sinkhorn(mu.reshape(-1), nu.reshape(-1), G, epsilon)
        else:
            gamma_tuta = uot(mu, nu, G, epsilon, rho = rho)
        # Line search for update
        CxC_tuta_minus_old = c1.dot(np.dot(gamma_tuta-gamma_old, c2))
        CxC_old = c1.dot(np.dot(gamma_old, c2))
        a = -alpha*np.sum( CxC_tuta_minus_old * ( gamma_tuta-gamma_old ) )
        b = np.sum( ( (1.0-alpha)*c+alpha*constC-2.0*alpha*CxC_old ) * ( gamma_tuta-gamma_old) )
        if a > 0:
            tau_update = min(1.0,max(0.0,-0.5*b/a))
        elif a + b < 0:
            tau_update = 1.0
        else:
            tau_update = 0.0
        gamma_new = (1.0-tau_update) * gamma_old + tau_update * gamma_tuta
        gamma_old = gamma_new
    return gamma_new

def ave(tau, u, u1):
    return tau * u + ( 1 - tau ) * u1

def lse(A):
    return np.log(np.sum(np.exp(A),axis=1)).reshape(-1,1)

def H(p):
    return -np.sum( p * np.log(p+1E-20)-1 )

def KL(h,p):
    return np.sum( h * np.log( h/p ) - h + p )

def KLd(u,p):
    return np.sum( p * ( np.exp(-u) - 1 ) )

def M(u,v,H1,H2,c,epsilon):
    y = -c + np.matmul(u.reshape(-1,1), H2.reshape(1,-1)) + \
        np.matmul(H1.reshape(-1,1), v.reshape(1,-1))
    return y/epsilon


def main():
    print(uot([0.5,0.5],[0.5,0.5],np.array([[0.0,1.0],[1.0,0.0]]),0.1,rho=np.Inf))
    a = np.ones(3)/3.0; b = a
    c = np.array([[1,4,4],[4,1,4],[4,4,1]], float)
    c1 = np.array([[0,10,0],[10,0,1],[0,1,0]], float)
    c2 = np.array([[0,10,0],[10,0,0],[0,0,0]], float)
    gamma = usot(a,b,c,c1,c2,0.5)
    print(gamma)

if __name__ == "__main__":
    main()
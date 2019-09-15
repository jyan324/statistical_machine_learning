from scipy.stats import multivariate_normal
rv_xGivenBlue = multivariate_normal([2.5,0], 
                                    [[2,1],[1,2]])
rv_xGivenRed = multivariate_normal([0,2.5], 
                                   [[2,1],[1,2]])
Bayes_pred = []
for pos in np.c_[x_mesh.ravel(),y_mesh.ravel()]:
    p_xGivenBlue = rv_xGivenBlue.pdf(pos);
    p_xGivenRed = rv_xGivenRed.pdf(pos);
    if (p_xGivenBlue > p_xGivenRed):
        Bayes_pred.append(1)
    else:
        Bayes_pred.append(0)
Bayes_pred = np.array(Bayes_pred)
Bayes_pred = Bayes_pred.reshape(x_mesh.shape)
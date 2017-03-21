import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(9)

def generate_data(n):
	y = []
	for w in np.random.sample(n):
		val = 2*np.random.randn()-10 if w < 0.5 else np.random.randn()
		val = 2*np.random.randn()+10 if w > 0.6 else val
		y.append(val)
	return y

#================== PARAMETERS =====================#
N = 500
R = 30 # length of range of data
# mu = N(xi, kappa-1)
xi = 0	
kappa = 1./(R**2)
# beta = f(g,h), sigma = Gamma(alpha, beta)
alpha = 2
g = 0.2
h = 10/(R**2)
# w = Dirichlet([delta] * k)
delta = 1 
#==================================================#

def allocation(k, W, MU, SIGMA):
	z = np.zeros(N)
	for i in range(N):
		z[i] = np.argmax([W[j]*norm.pdf(y[i], MU[j], np.sqrt(SIGMA[j])) for j in range(k)])
	return z

#================== INITIALIZATION =====================#
y = generate_data(N)
ki = 3
wi = np.array([0.3, 0.4, 0.3])
mui = np.array([-15.0, -3.0, 4.0])
sigmai = np.array([4.0,1.0, 8.0])
zi = allocation(ki, wi, mui, sigmai)
betai = np.random.gamma(g+ki*alpha, 1./(h + sum([1./sigmai[j] for j in range(ki)])))
init_state = {"k": ki, "W": wi, "MU": mui, "SIGMA": sigmai, "z": zi, "beta": betai}
#========================================================#

def sweep(state):
	k = state["k"]
	n = np.histogram(state["z"], bins=range(k+1))[0]
	#Step(a)
	w = np.random.dirichlet(n + delta,1)[0]

	#Step(b)
	mu = np.zeros(k)
	for j in range(k):
		i = np.where(state["z"]==j)[0]
		sigmainv = 1./(state["SIGMA"][j])
		muj = sigmainv*sum([y[index] + xi*kappa for index in i]) / (sigmainv*n[j] + kappa)
		sigmaj =  np.sqrt(1. / (sigmainv*n[j] + kappa))
		mu[j] = sigmaj * np.random.randn() + muj

	if not all(b >= a for a, b in zip(mu, mu[1:])):
		mu = state['MU']

	sigma = np.zeros(k)
	for j in range(k):
		i = np.where(state["z"]==j)[0]
		alphaj = alpha + 0.5*n[j]
		betaj = state["beta"] + 0.5*sum([(y[index] - mu[j])**2 for index in i])
		sigma[j] = 1./np.random.gamma(alphaj,1./betaj)
	
	#Step(c)
	z = allocation(k, w, mu, sigma)

	#Step(d)
	beta = np.random.gamma(g+k*alpha, 1./(h + sum([1./sigma[j] for j in range(k)])))

	#DONE
	return {"k": k, "W": w, "MU": mu, "SIGMA": sigma, "z": z, "beta": beta,}

# state = init_state
# for i in range(100):
# 	new_state = sweep(state)
# 	print i, "Weight: ", new_state["W"], "MU: ", new_state["MU"], "SIGMA: ", new_state["SIGMA"]
# 	state = new_state

state = init_state
finalw = None
finalmu = None
finalsigma = None
for i in range(50):
	print i
	new_state = sweep(state)
	if i == 25:
		finalw, finalmu, finalsigma = new_state["W"], new_state["MU"], new_state["SIGMA"]
	if i > 25:
		finalw += new_state["W"]
		finalmu += new_state["MU"]
		finalsigma += new_state["SIGMA"]
	state = new_state
finalw /= 25
finalmu /= 25
finalsigma /= 25
print finalw, finalmu, finalsigma
count, bins, ignored = plt.hist(y, 100, normed = True , alpha=0.75, color='cyan')

plt.plot(bins, sum([finalw[j]*norm.pdf(bins, finalmu[j], np.sqrt(finalsigma[j])) for j in range(state["k"])]), linewidth=3, color='magenta')
plt.show()
using Plots
using APL

l = 25000

V = zeros(l); V[1] = 10
for i = 2:l; V[i] = V[i-1]+(randn()*0.01); end

v = log.(V)
plot(1:l, v)
savefig("p_p.png")


k = 1
d = zeros(v)

for i = 1:l
	if i <= k
		d[i] = 0
	else
		d[i] = v[i]-v[i-k]
	end
end

m = median(d)
s = 2.5*ones(d)*std(d)

plot(1:l, hcat(d,s+m,m-s))
savefig("p_d.png")

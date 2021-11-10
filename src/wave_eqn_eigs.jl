using LinearAlgebra
using Arpack
using Gridap
using Gridap: ∇

function profiles(x)
  H  = 1.0 .+ 0.1.*cos.(x .- π)
  dH = 1.0 .- 0.1.*sin.(x .- π)
  s  = 1.0 ./ H ./ H
  ds = -2.0 .* dH ./ H ./ H ./ H
  S  = s .* H
  H, dH, s, ds, S
end

D = 2*π
order = 1
domain = (0,D)
partition = (128,)

model = CartesianDiscreteModel(domain, partition; isperiodic=(true,))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2*order)

V = FESpace(model, ReferenceFE(lagrangian, Float64, order), conformity=:H1)
Q = FESpace(model, ReferenceFE(lagrangian, Float64, order-1), conformity=:L2)
U = TrialFESpace(V)
P = TrialFESpace(Q)

h(x)  = profiles(x[1])[1]
dh(x) = profiles(x[1])[2]
s(x)  = profiles(x[1])[3]
ds(x) = profiles(x[1])[4]
S(x)  = profiles(x[1])[5]
f(x)  = 0.0*(h(x)-1.0)

mp(p,q) = ∫(q*p)dΩ
Mp = assemble_matrix(mp, P, Q)

mu(u,v) = ∫(v⋅u)dΩ
Mu = assemble_matrix(mu, U, V)

#md(u,q) = ∫(q*∇(u))dΩ
md(u,q) = ∫(q*((u->u[1])∘gradient(u)))dΩ
Md = assemble_matrix(md, U, Q)
Mg = transpose(Md)

MpInv = inv(Matrix(Mp))
MuInv = inv(Matrix(Mu))

muc(u,v) = ∫(f*u⋅v)dΩ
Muc = assemble_matrix(muc, U, V)

# wave eqn
A = MuInv*Mg*MpInv*Md# + MuInv*Muc*MuInv*Muc
ω1,v1 = eigs(A; nev=order*partition[1]-2)
ω1r = real(ω1)
ω1i = imag(ω1)

# flux form, S∈ L²(Ω)
mus(u,v) = ∫(v⋅u*s)dΩ
Mus = assemble_matrix(mus, U, V)
muh(u,v) = ∫(v⋅u*h)dΩ
Muh = assemble_matrix(muh, U, V)
muS(u,v) = ∫(v⋅u*S)dΩ
MuS = assemble_matrix(muS, U, V)

B = 0.5*MuInv*Mus*MuInv*Mg*MpInv*Md*MuInv*Muh + 0.5*MuInv*Mg*MpInv*Md*MuInv*MuS# + MuInv*Muc*MuInv*Muc
ω2,v2 = eigs(B; nev=order*partition[1]-2)
ω2r = real(ω2)
ω2i = imag(ω2)

# material form, s∈ H¹(Ω) 
#muds(u,v) = ∫(v⋅u*ds)dΩ
#Muds = assemble_matrix(muds, U, V)
muds(u,v) = ∫(s*v⋅((u->u[1])∘gradient(u))+s*u⋅((v->v[1])∘gradient(v)))dΩ
Muds = -1.0*assemble_matrix(muds, U, V)
mduu(u,v) = ∫(v⋅((u->u[1])∘gradient(u)))dΩ
Mduu = assemble_matrix(mduu, U, V)

C = MuInv*Mus*MuInv*Mg*MpInv*Md*MuInv*Muh + 0.5*MuInv*Muh*MuInv*Mduu*MuInv*Muds# + MuInv*Muc*MuInv*Muc
ω3,v3 = eigs(C; nev=order*partition[1]-2)
ω3r = real(ω3)
ω3i = imag(ω3)

# material form, s∈ L²(Ω)
msdpu(u,q) = ∫(s*((u->u[1])∘gradient(u*q)))dΩ
Msdpu = assemble_matrix(msdpu, U, Q)

D = MuInv*Mus*MuInv*Mg*MpInv*Md*MuInv*Muh - 0.5*MuInv*Mg*MpInv*Msdpu# + MuInv*Muc*MuInv*Muc
ω4,v4 = eigs(D; nev=order*partition[1]-2)
ω4r = real(ω4)
ω4i = imag(ω4)

plt = plot()
xq = LinRange(0, order*partition[1]-3, order*partition[1]-2)
xq = reverse(xq)

y1q = lazy_map(sqrt, ω1r)
y2q = lazy_map(sqrt, ω2r)
y3q = lazy_map(sqrt, ω3r)
y4q = lazy_map(sqrt, ω4r)

z1q = lazy_map(abs, ω1i)
z2q = lazy_map(abs, ω2i)
z3q = lazy_map(abs, ω3i)
z4q = lazy_map(abs, ω4i)

plot_imag=false
if(plot_imag)
plot!(plt, xq, 2*z1q, legend = true, seriestype=scatter)
plot!(plt, xq, 2*z2q, legend = true)
plot!(plt, xq, 2*z3q, legend = true)
plot!(plt, xq, 2*z4q, legend=:topleft)
else
plot!(plt, xq, xq, legend = false)
plot!(plt, xq, 2*y1q.-0*y1q[partition[1]-2], legend = true, seriestype=scatter)
plot!(plt, xq, 2*y2q.-0*y2q[partition[1]-2], legend = true)
plot!(plt, xq, 2*y3q.-0*y3q[partition[1]-2], legend = true)
plot!(plt, xq, 2*y4q.-0*y4q[partition[1]-2], legend=:topleft)
end

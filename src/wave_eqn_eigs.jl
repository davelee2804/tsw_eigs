using LinearAlgebra
using Plots
using Arpack
using Gridap
using Gridap: ∇

function profiles(x)
  α  = 0.5
  k  = 16
  H  = 1.0 .+ α.*cos.(k.*x .- π)
  dH = 1.0 .- k.*α.*sin.(k.*x .- π)
  e  = 1.0 ./ H ./ H
  de = -2.0 .* dH ./ H ./ H ./ H
  E  = e .* H
  f  = 0.0 .* sin.(x .- π)
  H, dH, e, de, E, f
end

D = 2*π
order = 0
domain = (0,D)
partition = (128,)

model = CartesianDiscreteModel(domain, partition; isperiodic=(true,))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2*(order+1))
Γ = SkeletonTriangulation(model)
dΓ = Measure(Γ, 0)
n_Γ = get_normal_vector(Γ)

S = FESpace(model, ReferenceFE(lagrangian, Float64, order+1), conformity=:H1)
V = FESpace(model, ReferenceFE(raviart_thomas, Float64, order), conformity=:HDiv)
Q = FESpace(model, ReferenceFE(lagrangian, Float64, order), conformity=:L2)
R = TrialFESpace(S)
U = TrialFESpace(V)
P = TrialFESpace(Q)

h(x)  = profiles(x[1])[1]
dh(x) = profiles(x[1])[2]
e(x)  = profiles(x[1])[3]
de(x) = profiles(x[1])[4]
E(x)  = profiles(x[1])[5]
f(x)  = profiles(x[1])[6]

h2 = interpolate_everywhere(h,Q)
e0 = interpolate_everywhere(e,S)
e2 = interpolate_everywhere(e,Q)
E2 = interpolate_everywhere(E,Q)

mp(p,q) = ∫(q*p)dΩ
Mp = assemble_matrix(mp, P, Q)
MpInv = inv(Matrix(Mp))

mu(u,v) = ∫(v⋅u)dΩ
Mu = assemble_matrix(mu, U, V)
MuInv = inv(Matrix(Mu))

mr(r,s) = ∫(s*r)dΩ
Mr = assemble_matrix(mr, R, S)
MrInv = inv(Matrix(Mr))

md(u,q) = ∫(q*(∇⋅u))dΩ
Md = assemble_matrix(md, U, Q)
#md(u,q) = ∫(q*((u->u[1])∘gradient(u)))dΩ
#Md = assemble_matrix(md, R, Q)
Mg = transpose(Md)
mgr(v,r) = ∫((∇⋅v)⋅r)dΩ
Mgr = assemble_matrix(mgr, R, V)

muc(u,v) = ∫(f*u⋅v)dΩ
Muc = assemble_matrix(muc, U, V)

# wave eqn
A = MuInv*Mg*MpInv*Md + MuInv*Muc*MuInv*Muc
ω1,v1 = eigs(A; nev=(order+1)*partition[1]-2)
ω1r = real(ω1)
ω1i = imag(ω1)

# flux form, E∈ L²(Ω)
mue(u,v) = ∫(v⋅u*e0)dΩ
Mue = assemble_matrix(mue, U, V)
muh(u,v) = ∫(v⋅u*h2)dΩ
Muh = assemble_matrix(muh, U, V)
muE(u,v) = ∫(v⋅u*E2)dΩ
MuE = assemble_matrix(muE, U, V)

B = 0.5*MuInv*Mue*MuInv*Mg*MpInv*Md*MuInv*Muh + 0.5*MuInv*Mg*MpInv*Md*MuInv*MuE + MuInv*Muc*MuInv*Muc
ω2,v2 = eigs(B; nev=(order+1)*partition[1]-2)
ω2r = real(ω2)
ω2i = imag(ω2)

# material form, e∈ H¹(Ω) 
#mude(u,v) = ∫(v⋅u*de)dΩ
#Mude = assemble_matrix(mude, U, V)
#mude(u,v) = ∫(e*v⋅((u->u[1])∘gradient(u))+e*u⋅((v->v[1])∘gradient(v)))dΩ
#mude(u,v) = ∫(∇(u⋅v))dΩ
#Mude = -1.0*assemble_matrix(mude, U, V)
#mduu(u,v) = ∫(v⋅((u->u[1])∘gradient(u)))dΩ
#Mduu = assemble_matrix(mduu, U, V)
mrude(u,s) = ∫(s*∇(e0)⋅u)dΩ
Mrude = assemble_matrix(mrude, U, S)

C = MuInv*Mue*MuInv*Mg*MpInv*Md*MuInv*Muh + 0.5*MuInv*Muh*MuInv*Mgr*MrInv*Mrude + MuInv*Muc*MuInv*Muc
ω3,v3 = eigs(C; nev=(order+1)*partition[1]-2)
ω3r = real(ω3)
ω3i = imag(ω3)

# material form, e∈ L²(Ω)
mue2(u,v) = ∫(v⋅u*e2)dΩ
Mue2 = assemble_matrix(mue2, U, V)
#medpu(u,q) = ∫(e*((u->u[1])∘gradient(u*q)))dΩ
#medpu(u,q) = ∫(e*q*((u->u[1])∘gradient(u)) + e*u*((q->q[1])∘gradient(q)))dΩ
medpu(u,q) = ∫(e2*q*(∇⋅u) + e2*u⋅∇(q))dΩ
Medpu = assemble_matrix(medpu, U, Q)
buqe(u,q) = ∫(mean(e2)*jump(q*u⋅n_Γ))dΓ
Buqe = assemble_matrix(buqe, U, Q)

D = MuInv*Mue2*MuInv*Mg*MpInv*Md*MuInv*Muh - 0.5*MuInv*Muh*MuInv*Mg*MpInv*Medpu + 0.5*MuInv*Muh*MuInv*Mg*MpInv*Buqe + MuInv*Muc*MuInv*Muc
ω4,v4 = eigs(D; nev=(order+1)*partition[1]-2)
ω4r = real(ω4)
ω4i = imag(ω4)

xq = LinRange(0, (order+1)*partition[1]-3, (order+1)*partition[1]-2)
xq = reverse(xq)

aω1r = lazy_map(abs, ω1r)
aω2r = lazy_map(abs, ω2r)
aω3r = lazy_map(abs, ω3r)
aω4r = lazy_map(abs, ω4r)
y1q = lazy_map(sqrt, aω1r)
y2q = lazy_map(sqrt, aω2r)
y3q = lazy_map(sqrt, aω3r)
y4q = lazy_map(sqrt, aω4r)

z1q = lazy_map(abs, ω1i)
z2q = lazy_map(abs, ω2i)
z3q = lazy_map(abs, ω3i)
z4q = lazy_map(abs, ω4i)
print(z1q,"\n")
print(z2q,"\n")
print(z3q,"\n")
print(z4q,"\n")
print(y1q[12-2]-1,"\n")
print(y2q[12-2]-1,"\n")
print(y3q[12-2]-1,"\n")
print(y4q[12-2]-1,"\n")

plot_imag=false
if(plot_imag)
plt2 = plot()
plot!(plt2, xq, 2*z1q, legend = true, seriestype=scatter)
plot!(plt2, xq, 2*z2q, legend = true)
plot!(plt2, xq, 2*z3q, legend = true)
plot!(plt2, xq, 2*z4q, legend=:topleft)
else
plt1 = plot()
#plot!(plt1, 0.5*xq, 0.5*xq, legend = false)
#plot!(plt1, 0.5*xq, y1q.-0*y1q[partition[1]-2], legend = true, seriestype=scatter)
#plot!(plt1, 0.5*xq, y2q.-0*y2q[partition[1]-2], legend = true)
#plot!(plt1, 0.5*xq, y3q.-0*y3q[partition[1]-2], legend = true)
#plot!(plt1, 0.5*xq, y4q.-0*y4q[partition[1]-2], legend=:topleft)
plot!(plt1, 0.5*xq, y1q-0.5*xq.-0*y1q[partition[1]-2].-1.0, legend = true, seriestype=scatter)
plot!(plt1, 0.5*xq, y2q-0.5*xq.-0*y2q[partition[1]-2].-1.0, legend = true, seriestype=scatter)
plot!(plt1, 0.5*xq, y3q-0.5*xq.-0*y3q[partition[1]-2].-1.0, legend = true)
plot!(plt1, 0.5*xq, y4q-0.5*xq.-0*y4q[partition[1]-2].-1.0, legend=:topleft)
display(plt1)
end

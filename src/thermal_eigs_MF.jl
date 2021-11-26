using LinearAlgebra
using Plots
using Arpack
using Gridap
using Gridap: ∇

function profiles(x)
  α  = 0.5
  k  = 1
  H  = 1.0 .+ α.*cos.(k.*x .- π)
  dH = 1.0 .- k.*α.*sin.(k.*x .- π)
  e  = 1.0 ./ H ./ H
  de = -2.0 .* dH ./ H ./ H ./ H
  E  = e .* H
  f  = 0.0 .* sin.(x .- π)
  a  = 2.0 .* dH .* dH ./ H ./ H ./ H
  H, dH, e, de, E, f, a
end

D = 2*π
order = 1
domain = (0,D)
ne = 8
scale = 4

partition = (scale*ne,)
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
a(x)  = profiles(x[1])[7]

h2 = interpolate_everywhere(h,Q)
e0 = interpolate_everywhere(e,S)
e2 = interpolate_everywhere(e,Q)
E2 = interpolate_everywhere(E,Q)
E0 = interpolate_everywhere(E,S)
a0 = interpolate_everywhere(a,S)

xq0 = LinRange(0, (order+1)*partition[1]-3, (order+1)*partition[1]-2)
xq0 = reverse(xq0)

mp(p,q) = ∫(q*p)dΩ
Mp = assemble_matrix(mp, P, Q)
MpInv = inv(Matrix(Mp))

mu(u,v) = ∫(v⋅u)dΩ
Mu = assemble_matrix(mu, U, V)
MuInv = inv(Matrix(Mu))

ms(r,s) = ∫(s*r)dΩ
Ms = assemble_matrix(ms, R, S)
MsInv = inv(Matrix(Ms))

md(u,q) = ∫(q*(∇⋅u))dΩ
D = assemble_matrix(md, U, Q)
G = transpose(D)

muc(u,v) = ∫(f*u⋅v)dΩ
Muc = assemble_matrix(muc, U, V)

# wave eqn
muE(u,v) = ∫(E0*u⋅v)dΩ
MuE = assemble_matrix(muE, U, V)
mua(u,v) = ∫(a0*u⋅v)dΩ
Mua = assemble_matrix(mua, U, V)
A = MuInv*MuE*MuInv*G*MpInv*D + MuInv*Mua #+ MuInv*Muc*MuInv*Muc
ω1,v1 = eigs(A; nev=(order+1)*partition[1]-2)
ω1r = real(ω1)
ω1i = imag(ω1)

partition = (ne,)
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
a(x)  = profiles(x[1])[7]

h2 = interpolate_everywhere(h,Q)
e0 = interpolate_everywhere(e,S)
e2 = interpolate_everywhere(e,Q)
E2 = interpolate_everywhere(E,Q)
E0 = interpolate_everywhere(E,S)

xq = LinRange(0, (order+1)*partition[1]-3, (order+1)*partition[1]-2)
xq = reverse(xq)

mp(p,q) = ∫(q*p)dΩ
Mp = assemble_matrix(mp, P, Q)
MpInv = inv(Matrix(Mp))

mu(u,v) = ∫(v⋅u)dΩ
Mu = assemble_matrix(mu, U, V)
MuInv = inv(Matrix(Mu))

ms(r,s) = ∫(s*r)dΩ
Ms = assemble_matrix(ms, R, S)
MsInv = inv(Matrix(Ms))

md(u,q) = ∫(q*(∇⋅u))dΩ
D = assemble_matrix(md, U, Q)
G = transpose(D)

muc(u,v) = ∫(f*u⋅v)dΩ
Muc = assemble_matrix(muc, U, V)

# material form, e∈ L²(Ω)
muh(u,v) = ∫(v⋅u*h2)dΩ
Muh = assemble_matrix(muh, U, V)
mue2(u,v) = ∫(v⋅u*e2)dΩ
Mue2 = assemble_matrix(mue2, U, V)
Dh = D*MuInv*Muh

gh(p,v) = ∫(-1.0*(∇⋅v)*e2*p)dΩ
Gh = assemble_matrix(gh, P, V)

ds(u,q) = ∫(-1.0*(∇⋅(q*u))*e2)dΩ + ∫(mean(e2)*jump((q*u)⋅n_Γ))dΓ
Ds = assemble_matrix(ds, U, Q)

gs(p,v) = ∫(0.5*(∇⋅(h2*v))*p)dΩ - ∫(mean(h2)*jump((p*v)⋅n_Γ))dΓ
Gs = assemble_matrix(gs, P, V)

A2 = MuInv*Gh*MpInv*Dh + MuInv*Gs*MpInv*Ds
ω2,v2 = eigs(A2; nev=(order+1)*partition[1]-2)
ω2r = real(ω2)
ω2i = imag(ω2)

# material form, e∈ H¹(Ω)
muh(u,v) = ∫(v⋅u*h2)dΩ
Muh = assemble_matrix(muh, U, V)
mue0(u,v) = ∫(v⋅u*e0)dΩ
Mue0 = assemble_matrix(mue0, U, V)
Dh = D*MuInv*Muh

gh(p,v) = ∫(-1.0*(∇⋅v)*e0*p)dΩ
Gh = assemble_matrix(gh, P, V)

ds(u,s) = ∫(-1.0*(∇⋅(s*u))*e0)dΩ + ∫(jump((e0*s*u)⋅n_Γ))dΓ
Ds = assemble_matrix(ds, U, S)

gs(r,v) = ∫(0.5*(∇⋅(h2*v))*r)dΩ - ∫(mean(h2)*jump((r*v)⋅n_Γ))dΓ
Gs = assemble_matrix(gs, R, V)

A3 = MuInv*Gh*MpInv*Dh + MuInv*Gs*MsInv*Ds
ω3,v3 = eigs(A3; nev=(order+1)*partition[1]-2)
ω3r = real(ω3)
ω3i = imag(ω3)

aω1r = lazy_map(abs, ω1r)
aω2r = lazy_map(abs, ω2r)
aω3r = lazy_map(abs, ω3r)
y1q = lazy_map(sqrt, aω1r)
y2q = lazy_map(sqrt, aω2r)
y3q = lazy_map(sqrt, aω3r)
z1q = lazy_map(abs, ω1i)
z2q = lazy_map(abs, ω2i)
z3q = lazy_map(abs, ω3i)

#print(z1q,"\n")
#print(z2q,"\n")
#print(z3q,"\n")
#print(z4q,"\n")
xq0=xq0[(scale-1)*(order+1)*ne+1:scale*(order+1)*ne-2]
y1q=y1q[(scale-1)*(order+1)*ne+1:scale*(order+1)*ne-2]
print(xq0,"\n")
print(xq,"\n")

plt1 = plot()
#plot!(plt1, 0.5*xq, 0.5*xq, legend = true, label="ω = k")
 
#plot!(plt1, 0.5*xq0, y1q, legend = true, label="Wave eqn.")
#plot!(plt1, 0.5*xq, y2q, legend = :topleft, label="s∈ L²(Ω), Integration by parts", seriestype=:scatter)
#plot!(plt1, 0.5*xq, y3q, legend = :topleft, label="s∈ H¹(Ω), Integration by parts")

plot!(plt1, 0.5*xq, y2q.-y1q, legend = :topleft, label="s∈ L²(Ω), Integration by parts", seriestype=:scatter)
plot!(plt1, 0.5*xq, y3q.-y1q, legend = :topleft, label="s∈ H¹(Ω), Integration by parts")
 
savefig("thermal_eigs_mf.png")
display(plt1)


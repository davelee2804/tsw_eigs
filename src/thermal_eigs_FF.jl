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
order = 1
domain = (0,D)
ne = 64
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

h2 = interpolate_everywhere(h,Q)
e0 = interpolate_everywhere(e,S)
e2 = interpolate_everywhere(e,Q)
E2 = interpolate_everywhere(E,Q)
E0 = interpolate_everywhere(E,S)

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
A = MuInv*MuE*MuInv*G*MpInv*D #+ MuInv*Muc*MuInv*Muc
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

# Flux form, E∈ L²(Ω)
muh(u,v) = ∫(v⋅u*h2)dΩ
Muh = assemble_matrix(muh, U, V)
mue2(u,v) = ∫(v⋅u*e0)dΩ
Mue2 = assemble_matrix(mue2, U, V)

Dh = D*MuInv*Muh

# projection in the poisson bracket
Ge2 = Mue2*MuInv*G
De2 = D*MuInv*Mue2*MuInv*Muh
A2 = 0.5*MuInv*Ge2*MpInv*Dh + 0.5*MuInv*G*MpInv*De2
ω2,v2 = eigs(A2; nev=(order+1)*partition[1]-2)
ω2r = real(ω2)
ω2i = imag(ω2)

# integration by parts
ge2(p,v) = ∫((∇⋅(e0*v))*p)dΩ
Ge2 = assemble_matrix(ge2, P, V)
# non-energy conserving divergence
Ge2T = transpose(Ge2)
De2 = Ge2T*MuInv*Muh
#c1(p,v) = ∫(jump(e0*p*v⋅n_Γ))dΓ
c1(p,v) = ∫(mean(p)*jump((e0*v)⋅n_Γ))dΓ
C1 = assemble_matrix(c1, P, V)
Ge2 = Ge2 - C1
A3 = 0.5*MuInv*Ge2*MpInv*Dh + 0.5*MuInv*G*MpInv*De2
ω3,v3 = eigs(A3; nev=(order+1)*partition[1]-2)
ω3r = real(ω3)
ω3i = imag(ω3)

# integration by parts
ge2(p,v) = ∫((∇⋅(e0*v))*p)dΩ
Ge2 = assemble_matrix(ge2, P, V)
# energy conserving divergence
#c1(p,v) = ∫(jump(e0*p*v⋅n_Γ))dΓ
c1(p,v) = ∫(mean(p)*jump((e0*v)⋅n_Γ))dΓ
C1 = assemble_matrix(c1, P, V)
Ge2 = Ge2 - C1
Ge2T = transpose(Ge2)
De2 = Ge2T*MuInv*Muh
A4 = 0.5*MuInv*Ge2*MpInv*Dh + 0.5*MuInv*G*MpInv*De2
ω4,v4 = eigs(A4; nev=(order+1)*partition[1]-2)
ω4r = real(ω4)
ω4i = imag(ω4)

# Flux form, E∈ H¹(Ω)
msp(p,s) = ∫(s*p)dΩ
Msp = assemble_matrix(msp, P, S)
Mps = transpose(Msp)

muh(u,v) = ∫(v⋅u*h2)dΩ
Muh = assemble_matrix(muh, U, V)
mue2(u,v) = ∫(v⋅u*e2)dΩ
Mue2 = assemble_matrix(mue2, U, V)

Dh = D*MuInv*Muh

mue0(u,v) = ∫(v⋅u*e2)dΩ
Mue0 = assemble_matrix(mue0, U, V)
mgs(v,r) = ∫((∇⋅v)⋅r)dΩ
Gs = assemble_matrix(mgs, R, V)
Ds = transpose(Gs)

# projections in the Poisson bracket
Gse0 = Mue0*MuInv*Gs
Dse0 = Ds*MuInv*Mue0*MuInv*Muh
FF5 = 0.5*MuInv*Gse0*MsInv*Msp*MpInv*Dh + 0.5*MuInv*G*MpInv*Mps*MsInv*Dse0
ω5,v5 = eigs(FF5; nev=(order+1)*partition[1]-2)
ω5r = real(ω5)
ω5i = imag(ω5)

# energy conserving divergence
de0(u,s) = ∫(-1.0*e2*u⋅∇(s))dΩ + ∫(mean(e2)*jump((s*u)⋅n_Γ))dΓ
Gse0T = assemble_matrix(de0, U, S)
Gse0 = transpose(Gse0T)
Dse0 = Gse0T*MuInv*Muh
FF6 = 0.5*MuInv*Gse0*MsInv*Msp*MpInv*Dh + 0.5*MuInv*G*MpInv*Mps*MsInv*Dse0
ω6,v6 = eigs(FF6; nev=(order+1)*partition[1]-2)
ω6r = real(ω6)
ω6i = imag(ω6)

# non-energy conserving divergence
de0(u,s) = ∫(-1.0*e2*u⋅∇(s))dΩ
Gse0T = assemble_matrix(de0, U, S)
Gse0 = transpose(Gse0T)
FF7 = 0.5*MuInv*Gse0*MsInv*Msp*MpInv*Dh + 0.5*MuInv*G*MpInv*Mps*MsInv*Dse0
ω7,v7 = eigs(FF7; nev=(order+1)*partition[1]-2)
ω7r = real(ω7)
ω7i = imag(ω7)

aω1r = lazy_map(abs, ω1r)
aω2r = lazy_map(abs, ω2r)
aω3r = lazy_map(abs, ω3r)
aω4r = lazy_map(abs, ω4r)
aω5r = lazy_map(abs, ω5r)
aω6r = lazy_map(abs, ω6r)
aω7r = lazy_map(abs, ω7r)
y1q = lazy_map(sqrt, aω1r)
y2q = lazy_map(sqrt, aω2r)
y3q = lazy_map(sqrt, aω3r)
y4q = lazy_map(sqrt, aω4r)
y5q = lazy_map(sqrt, aω5r)
y6q = lazy_map(sqrt, aω6r)
y7q = lazy_map(sqrt, aω7r)
z1q = lazy_map(abs, ω1i)
z2q = lazy_map(abs, ω2i)
z3q = lazy_map(abs, ω3i)
z4q = lazy_map(abs, ω4i)
z5q = lazy_map(abs, ω5i)
z6q = lazy_map(abs, ω6i)
z7q = lazy_map(abs, ω7i)

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
#plot!(plt1, 0.5*xq, y4q, legend = :topleft, label="S∈ L²(Ω), Integration by parts", seriestype=:scatter)
#plot!(plt1, 0.5*xq, y2q, legend = true, label="S∈ L²(Ω), Projection")
#plot!(plt1, 0.5*xq, y3q, legend = true, label="IPB, non energy conserving")
#plot!(plt1, 0.5*xq, y6q, legend = :topleft, label="S∈ H¹(Ω), Integration by parts", seriestype=:scatter)
#plot!(plt1, 0.5*xq, y5q, legend = :topleft, label="S∈ H¹(Ω), Projection")
#plot!(plt1, 0.5*xq, y7q, legend = :topleft, label="S∈ H¹(Ω), IBP, non-energy conserving")

plot!(plt1, 0.5*xq, y4q.-y1q, legend = :topleft, label="S∈ L²(Ω), Integration by parts", seriestype=:scatter)
plot!(plt1, 0.5*xq, y2q.-y1q, legend = true, label="S∈ L²(Ω), Projection")
#plot!(plt1, 0.5*xq, y3q.-y1q, legend = true, label="S∈ L²(Ω), IPB, non energy conserving",seriestype=:scatter)
plot!(plt1, 0.5*xq, y6q.-y1q, legend = :topleft, label="S∈ H¹(Ω), Integration by parts", seriestype=:scatter)
plot!(plt1, 0.5*xq, y5q.-y1q, legend = :topleft, label="S∈ H¹(Ω), Projection")
#plot!(plt1, 0.5*xq, y7q.-y1q, legend = true, label="S∈ H¹(Ω), IPB, non energy conserving",seriestype=:scatter)
 
#plot!(plt1, 0.5*xq, 0.5*xq)
#plot!(plt1, 0.5*xq, y1q.-1.0)
#plot!(plt1, 0.5*xq, y2q.-1.0)
#plot!(plt1, 0.5*xq, y3q.-1.0)
#plot!(plt1, 0.5*xq, y4q.-1.0, legend=false)
savefig("thermal_eigs_ff.png")
display(plt1)


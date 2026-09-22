% TINKLAS_VIENAME_FAILE Pateikto 2-3-2-3-2 tinklo mokymas.
% Visas kodas viename faile: atidarykite ji MATLAB ir spauskite Run.
% Papildomu .m failu nereikia; vietines funkcijos pateiktos apacioje.
% Nereikia Deep Learning Toolbox: naudojamos musu paciu BP formules.
clear; clc;

% Patikriname atsaka ir visu 29 koeficientu atnaujinima.
% Jei patikros nereikia, pakeiskite true i false.
vykdyti_patikra = true;
if vykdyti_patikra
    patikrinti_tinkla();
end

% PAVYZDINIAI duomenys, nes uzduotyje konkretus duomenys nepateikti.
% Kiekvienas stulpelis yra atskiras pavyzdys: X(:,k) = [x1; x2].
% Pirmas norimas isejimas - XOR, antras - jo priesingybe (XNOR).
% Gavus destytojo duomenis, pakeiskite X ir D.
X = [0 0 1 1; ...
     0 1 0 1];
D = [0 1 1 0; ...
     1 0 0 1];

tinklas = inicijuoti_tinkla(1);
eta = 0.1;
epochu = 8000;
N = size(X, 2);
assert(size(X,1) == 2 && isequal(size(D), [2 N]), ...
    'X ir D turi tureti po 2 eilutes ir vienoda stulpeliu skaiciu.');

Y_pries = zeros(2, N);
for k = 1:N
    Y_pries(:,k) = tinklo_atsakas(tinklas, X(:,k));
end

% Mokymas po viena pavyzdi (online / stochastic BP).
% Viena epocha - visu pateiktu pavyzdziu perziura.
for epocha = 1:epochu
    tvarka = randperm(N);
    for k = tvarka
        tinklas = mokymo_zingsnis(tinklas, X(:,k), D(:,k), eta);
    end
end

% Po mokymo atsakus skaiciuojame is naujo su GALUTINIAIS svoriais.
Y_po = zeros(2, N);
for k = 1:N
    Y_po(:,k) = tinklo_atsakas(tinklas, X(:,k));
end
E_po = D - Y_po;  % Tik standartine klaida; tikslo funkcijos nera.

disp('Pries mokyma (eilutes: y1, y2; stulpeliai: pavyzdziai):');
disp(Y_pries);
disp('Po mokymo:');
disp(Y_po);
disp('Norimi isejimai D:');
disp(D);
disp('Klaidos E = D - Y:');
disp(E_po);

% Isvedame visus koeficientus. Matricu nuliai, kuriuos nurodo M,
% yra neegzistuojancios jungtys, o ne papildomi mokomi svoriai.
for l = 1:4
    fprintf('\n%d sluoksnis: W{%d}\n', l, l);
    disp(tinklas.W{l});
    fprintf('beta{%d}\n', l);
    disp(tinklas.beta{l});
end

% Pavyzdys vienai naujai iejimu porai:
x_naujas = [0; 1];
y_naujas = tinklo_atsakas(tinklas, x_naujas);
fprintf('\nKai x=[0;1], y=[%.6f; %.6f]\n', y_naujas(1), y_naujas(2));



%% VIETINES FUNKCIJOS: palikite tame paciame faile


function tinklas = inicijuoti_tinkla(sekla)
% INICIJUOTI_TINKLA Sukuria pateikto 2-3-2-3-2 tinklo koeficientus.
% W{l}(j,i): svoris IS ankstesnio sluoksnio i-ojo neurono
% I l-ojo sluoksnio j-aji neurona. Beta{l}(j): jo bias.
% Neuronai numeruojami is virsaus i apacia.
% Nuliai kaukese zymi neegzistuojancias jungtis; jos nemokomos.

if nargin < 1
    sekla = 1;
end
rng(sekla, 'twister');

tinklas.M = { ...
    [1 1; 1 1; 1 1], ...        % 6 jungtys: iejimai -> 1 sluoksnis
    [1 1 0; 0 1 1], ...         % 4 jungtys: 1 -> 2 sluoksnis
    [1 0; 1 1; 0 1], ...        % 4 jungtys: 2 -> 3 sluoksnis
    [1 1 1; 1 1 0]};            % 5 jungtys: 3 -> 4 sluoksnis

tinklas.W = cell(1, 4);
tinklas.beta = cell(1, 4);
for l = 1:4
    % Skirtingi atsitiktiniai svoriai panaikina neuronu simetrija.
    tinklas.W{l} = randn(size(tinklas.M{l})) .* tinklas.M{l};
    tinklas.beta{l} = zeros(size(tinklas.M{l}, 1), 1);
end
end


function [y, busena] = tinklo_atsakas(tinklas, x)
% TINKLO_ATSAKAS Apskaiciuoja atsaka vienai iejimu porai.
% x = [x1; x2]. y = [y1; y2].
% busena.v{l} saugo svertines sumas, busena.y{l} - neuronu isejimus.
% Si busena reikalinga to paties mokymo zingsnio atgaliniam sklidimui.

validateattributes(x, {'numeric'}, {'real', 'finite', 'vector', 'numel', 2});
x = double(x(:));
W = tinklas.W;
b = tinklas.beta;
v = cell(1, 4);
a = cell(1, 4);

% 1 sluoksnis: abu iejimai eina i visus tris neuronus; sigmoide.
v{1} = [ ...
    W{1}(1,1)*x(1) + W{1}(1,2)*x(2) + b{1}(1); ...
    W{1}(2,1)*x(1) + W{1}(2,2)*x(2) + b{1}(2); ...
    W{1}(3,1)*x(1) + W{1}(3,2)*x(2) + b{1}(3)];
a{1} = sigmoide(v{1});

% 2 sluoksnis: tik keturios brezinio jungtys; phi2(v) = v.
v{2} = [ ...
    W{2}(1,1)*a{1}(1) + W{2}(1,2)*a{1}(2) + b{2}(1); ...
    W{2}(2,2)*a{1}(2) + W{2}(2,3)*a{1}(3) + b{2}(2)];
a{2} = v{2};

% 3 sluoksnis: tik keturios brezinio jungtys; phi3(v) = v.
v{3} = [ ...
    W{3}(1,1)*a{2}(1) + b{3}(1); ...
    W{3}(2,1)*a{2}(1) + W{3}(2,2)*a{2}(2) + b{3}(2); ...
    W{3}(3,2)*a{2}(2) + b{3}(3)];
a{3} = v{3};

% 4 sluoksnis: pirmas isejimas gauna 3 signalus, antras - tik 2.
% Jungties W{4}(2,3) NERA. Abieju isejimu aktyvacija - sigmoide.
v{4} = [ ...
    W{4}(1,1)*a{3}(1) + W{4}(1,2)*a{3}(2) + ...
    W{4}(1,3)*a{3}(3) + b{4}(1); ...
    W{4}(2,1)*a{3}(1) + W{4}(2,2)*a{3}(2) + b{4}(2)];
a{4} = sigmoide(v{4});

y = a{4};
busena.x = x;
busena.v = v;
busena.y = a;
end

function y = sigmoide(v)
% Ta pati 1/(1+exp(-v)) formule, uzrasyta vengiant exp perpildymo.
y = zeros(size(v));
teigiami = v >= 0;
y(teigiami) = 1 ./ (1 + exp(-v(teigiami)));
ev = exp(v(~teigiami));
y(~teigiami) = ev ./ (1 + ev);
end


function [tinklas, y, e, delta] = mokymo_zingsnis(tinklas, x, d, eta)
% MOKYMO_ZINGSNIS Vieno pavyzdzio atsakas ir BP koeficientu atnaujinimas.
% d = [d1; d2] - norimi isejimai; eta - teigiamas mokymosi zingsnis.
% Klaida e = d-y. Tikslo funkcija cia neskaiciuojama.
% Grazinami y ir e yra PRIES sio zingsnio koeficientu atnaujinima.

validateattributes(d, {'numeric'}, {'real', 'finite', 'vector', 'numel', 2});
validateattributes(eta, {'numeric'}, {'real', 'finite', 'scalar', 'positive'});
d = double(d(:));
[y, s] = tinklo_atsakas(tinklas, x);
e = d - y;

% Kopija W saugo SENUOSIUS svorius. Visi delta turi buti apskaiciuoti
% su tais paciais svoriais, kurie naudoti sklidimui pirmyn.
W = tinklas.W;
a = s.y;
delta = cell(1, 4);

% Isejimo sluoksnis: delta_j = e_j * y_j * (1-y_j).
delta{4} = e .* a{4} .* (1 - a{4});

% 3 sluoksnis: aktyvacijos isvestine = 1.
% Kiekvienas neuronas sumuoja tik savo pasiekiamu neuronu gradientus.
delta{3} = [ ...
    W{4}(1,1)*delta{4}(1) + W{4}(2,1)*delta{4}(2); ...
    W{4}(1,2)*delta{4}(1) + W{4}(2,2)*delta{4}(2); ...
    W{4}(1,3)*delta{4}(1)];

% 2 sluoksnis: aktyvacijos isvestine = 1.
delta{2} = [ ...
    W{3}(1,1)*delta{3}(1) + W{3}(2,1)*delta{3}(2); ...
    W{3}(2,2)*delta{3}(2) + W{3}(3,2)*delta{3}(3)];

% 1 sluoksnis: prie svertines gradientu sumos pridedamas DAUGIKLIS
% phi1'(v_j) = y_j*(1-y_j), nes aktyvacija yra sigmoide.
suma = [ ...
    W{2}(1,1)*delta{2}(1); ...
    W{2}(1,2)*delta{2}(1) + W{2}(2,2)*delta{2}(2); ...
    W{2}(2,3)*delta{2}(2)];
delta{1} = a{1} .* (1 - a{1}) .* suma;

% Tik apskaiciave VISUS delta atnaujiname koeficientus.
% delta{l} * iejimas.' sudaro matrica, kurios (j,i) elementas
% yra delta_j * y_i. Kaukė M palieka tik brezinio jungtis.
for l = 1:4
    if l == 1
        iejimas = s.x;
    else
        iejimas = a{l-1};
    end
    pokytis = eta * (delta{l} * iejimas.');
    tinklas.W{l} = (W{l} + pokytis) .* tinklas.M{l};

    % Bias yra svoris prie pastovaus iejimo 1: beta += eta*delta.
    tinklas.beta{l} = tinklas.beta{l} + eta * delta{l};
end
end


function patikrinti_tinkla()
% PATIKRINTI_TINKLA Patikrina struktura, atsaka ir visu koeficientu BP.
% Tikrinant BP lyginami skaitiniai ISEJIMU pokyciai ir BP pokyciai.
% Jokia tikslo funkcija nei mokymui, nei siam testui neapibreziama.
t = inicijuoti_tinkla(17);
x = [0.2; -0.7];
d = [0.8; 0.1];
eta = 0.05;
[naujas, y, e, delta] = mokymo_zingsnis(t, x, d, eta);
assert(sum(cellfun(@nnz, t.M)) == 19);
assert(sum(cellfun(@numel, t.beta)) == 10);

% Nepriklausomas matricinis atsako perskaiciavimas.
a = x;
for l = 1:4
    a = t.W{l}*a + t.beta{l};
    if l == 1 || l == 4
        a = 1 ./ (1 + exp(-a));
    end
end
assert(max(abs(y-a)) < 1e-12, 'Nesutampa sklidimas pirmyn.');
assert(max(abs(e-(d-y))) < 1e-12);

% e laikome fiksuota. Tikriname kiekviena svori ir bias:
% atnaujinimas/eta turi sutapti su e.' * (dy/dkoeficientas).
h = 1e-5;
didziausias_skirtumas = 0;
for l = 1:4
    assert(all(naujas.W{l}(t.M{l} == 0) == 0), 'Sukurta neegzistuojanti jungtis.');
    for idx = find(t.M{l}).'
        plius = t; minus = t;
        plius.W{l}(idx) = plius.W{l}(idx) + h;
        minus.W{l}(idx) = minus.W{l}(idx) - h;
        dy = (tinklo_atsakas(plius,x) - tinklo_atsakas(minus,x))/(2*h);
        tiketinas = e.' * dy;
        gautas = (naujas.W{l}(idx) - t.W{l}(idx))/eta;
        didziausias_skirtumas = max(didziausias_skirtumas, abs(tiketinas-gautas));
    end
    for j = 1:numel(t.beta{l})
        plius = t; minus = t;
        plius.beta{l}(j) = plius.beta{l}(j) + h;
        minus.beta{l}(j) = minus.beta{l}(j) - h;
        dy = (tinklo_atsakas(plius,x) - tinklo_atsakas(minus,x))/(2*h);
        tiketinas = e.' * dy;
        gautas = (naujas.beta{l}(j) - t.beta{l}(j))/eta;
        didziausias_skirtumas = max(didziausias_skirtumas, abs(tiketinas-gautas));
        assert(abs(gautas-delta{l}(j)) < 1e-12);
    end
end
assert(didziausias_skirtumas < 1e-7, 'BP neatitinka skaitiniu isvestiniu.');
fprintf('Patikrinti 19 svoriu, 10 bias ir atsakas. Didziausias BP skirtumas: %.3g\n', ...
    didziausias_skirtumas);
end

% OUTPUT ADJUSTMENTS
pltall = 0; pltbest = 0;
 
% PROBLEM SPECS
Nm = 100; Nt = 100; No = 25; % number of agents, number of targets, number of obstacles.
drange = 5; crange = 2; % target collection and collision radii
bounds = [100, 100, 10]; % target and obstacle first octant dimensions
 
% ARRANGE OBJECTS INITIALLY
Tx = 2*(rand(Nt, 3) - .5) .* bounds; % random target position, same for all strings.
Ox = 2*(rand(No, 3) - .5) .* bounds; % random obstacle position, same for all strings.
 
%%%%%%%%%%%%% POSITION DRONES INITIALLY %%%%%%%%%%%%%%%%%
rect = closestFactors(Nm); % find factors to put agents into a nice grid. Arbitrary.
xg = 0:rect(1)-1; yg = (0:rect(2)-1) - (rect(2)-1)/2;
[Xg, Yg] = meshgrid(xg, yg);
x_o = bounds .* [-1*ones(Nm, 1), zeros(Nm, 2)] - max(crange, drange) - 3*crange*[Xg(:) , Yg(:) , zeros(Nm, 1)]; % Place swarm in an orderly structure to begin with, outside of target / obstacle space.
%%%%%%%%%%%%% POSITION DRONES INITIALLY %%%%%%%%%%%%%%%%%
 
% GA SPECS
scale = 2*ones(1, 15); % max - min for each design variable
offset = scale/2; % mean value for each design variable
ma = 2; % mutation amplitude, always centered on .5. 1 = no mutations.
sa = 3; % scale amplitude, multiplied by standard deviation when refining search space.
 
% DRONE DYNAMICS SPECS
Aeff = 1; % agent effective area, m^2.
Cd = 1; % agent coefficient of drag. 1.5 for air agent
rho = 1.225; % fluid density, kg/m^3
m = 10; % agent mass, kg.
F = 200; % total thrust, newtons.
vmax = sqrt(2*F / (rho*Cd*Aeff))
 
% TIME INTEGRATION SPECS
tf = 60; % ceil(4*sum(bounds)/vmax) % final time sufficient to traverse half of all domain edges (bounds defines an octact)
dt_rough = .2; % round(min(crange, drange)/(.25*vmax), 2, 'significant'); % time step, seconds to reduce odds of misses. 
nt = tf / dt_rough; nt = ceil(nt); dt = tf / nt; 
int = dt*5; % plotting intervals
 
% GENETIC ALGORITHM SPECS
w1 = 70; w2 = 10; w3 = 20; % cost function weights. target mapping, time use, preservation of agents.
 
S = 20; % number of strings per generation
parents = 2; children = parents; % breeding population parameters
G = 300; % total generations
reset = 10; % number of generations between recentering steps
TOL = 5; % allowable cost function value
 
% STORAGE ARRAYS
comp_time = zeros(G, S); % array to store computation time for each design string. 
PI = 100*ones(G, S); % store overall performance for all strings.
LO = PI; MO = PI; TO = PI; % store individual performance metrics
 
vc = []; % array to store velocities that had to be corrected
Orig = zeros(G, S); % place tto store history of where best design in each generation came from
scaleO = scale; offsetO = offset; % store scale and offset information to understand recentering history
 
% Generational storage arrays. All initialized with worst possible value.
Pi = 100*ones(1, S); % cost function array
Mg = 100*ones(1, S); % loss fraction array
Lg = 100*ones(1, S); % unmapped target fraction
Tg = 100*ones(1, S); % used time fraction
 
L = (rand(S , 15) - .5) .* scale + offset; % design strings.
g = 0; 
 
while g <= G % FOR EACH GENERATION 
    g = g + 1;
    for s = 1:S % FOR EACH STRING
        tic
        if (g == 1 || s > parents) || (pltbest && s == 1) % if this string has not been checked before OR if I want to plot the best one.
            % extract genetic string values for clarity
            wt1 = L(s, 1); wt2 = L(s, 2);
            a1 = L(s, 3); a2 = L(s, 4);
            wo1 = L(s, 5); wo2 = L(s, 6);
            b1 = L(s, 7); b2 = L(s, 8);
            wm1 = L(s, 9); wm2 = L(s, 10);
            c1 = L(s, 11); c2 = L(s, 12);
            Wmt = L(s, 13); Wmo = L(s, 14); Wmm = L(s, 15);
            
            % RESET PROBLEM STATE
            t = 0; thresh = 0; % reset time, plotting time, status of problem. 
            x = x_o; % initial agent positions.
            v = zeros(size(x)); % initial agent velocities all zero
            r_alive = true(Nm, 1); T_alive = true(Nt, 1); % everything is alive
            Lg = zeros(1, S); Mg = zeros(1, S); Tg = zeros(1, S); 
            while t < tf && any(r_alive) && any(T_alive) % TIME LOOP
                t = t + dt; % increment time
 
                N_mt = permute(Tx, [1, 3, 2]) - permute(x, [3, 1, 2]); % all vectors mt
                d_mt = vecnorm(N_mt, 2, 3); % all distances mt
                T_alive(any(d_mt < drange, 2)) = false; % if any target is close to a drone, it's dead
                n_mt = N_mt .* (1 ./ d_mt); 
                Nimt = permute(sum((wt1*exp(-a1*d_mt) - wt2*exp(-a2*d_mt)).*n_mt.*T_alive , 1), [2, 3, 1]); 
                Nimt(isnan(Nimt)) = 0; 
                
                N_mm = permute(x, [1, 3, 2]) - permute(x, [3, 1, 2]); 
                d_mm = vecnorm(N_mm, 2, 3) + diag(diag(Inf*eye(Nm))) + eps; % all distances mm
                r_alive(any(abs(x) >= (bounds + 50), 2)) = false; % remove stray drones
                r_alive(any(d_mm < crange, 1)) = false; r_alive(any(d_mm < crange, 2)) = false; 
                n_mm = N_mm .* (1 ./ d_mm); 
                Nimm = permute(sum((wm1*exp(-b1*d_mm) - wm2*exp(-b2*d_mm)).*n_mm.*r_alive , 1), [2, 3, 1]);
                Nimm(isnan(Nimm)) = 0; 
                
                N_mo = permute(Ox, [1, 3, 2]) - permute(x, [3, 1, 2]); 
                d_mo = vecnorm(N_mo, 2, 3); % all distances mo
                assert(all(d_mo > 0, 'all'))
                r_alive(any(d_mo < crange, 1)) = 0; 
                n_mo = N_mo .* (1 ./ d_mo); 
                Nimo = permute(sum((wo1*exp(-c1*d_mo) - wo2*exp(-c2*d_mo)).*n_mo , 1), [2, 3, 1]); 
                Nimo(isnan(Nimo)) = 0; 
                
                Nitot = Wmt * Nimt + Wmo * Nimo + Wmm * Nimm; % total interaction
                Nitot(isnan(Nitot)) = 0; 
                
                Fd = (1/2)*rho*Cd*Aeff*vecnorm(v, 2, 2).*v; % drag force
                Phi = F*Nitot ./ (vecnorm(Nitot, 2, 2) + eps) - Fd; % maximum force in desired direction, plus drag force plus gravity
                v(not(r_alive), :) = 0; 
                x = x + dt*v; % update position
                v(r_alive, :) = v(r_alive, :) + (dt/m)*Phi(r_alive, :); % update velocity
                
                % PLOT DOMAIN
                if (pltall || (pltbest && g > 1 && s == 1)) && t > thresh
                    thresh = thresh + int; figure(1); clf; plotDomain(x(r_alive, :), Tx(T_alive, :), Ox, g, s, t); pause(.001);
                end
            end % end time
            % compute cost function
            Mg(s) = nnz(T_alive) / Nt; % fraction of remaining targets.
            Tg(s) = t/tf; % fraction of time used
            Lg(s) = (Nm - nnz(r_alive))/Nm; % fraction of agents lost
            Pi(s) = w1*Mg(s) + w2*Tg(s) + w3*Lg(s);
        end
        myProgressBar(sum(comp_time(:)), s + S*(g - 1), S*G, ['Mapped: ' num2str(Nt - nnz(T_alive)) '/' num2str(Nt) ', Swarm: ' num2str(nnz(r_alive)) '/' num2str(Nm) ', g: ' num2str(g) ' s: ' num2str(s) ' t: ' num2str(t)])
        comp_time(g, s) = toc;
    end % end string
    
    % SORT
    [Pi, I] = sort(Pi); % rank the strings
    Orig(g, :) = I; % save sorting order for family tree
    L = L(I, :); % sort design strings
    PI(g, :) = Pi; 
    LO(g, :) = Lg(I);
    MO(g, :) = Mg(I);
    TO(g, :) = Tg(I);
    
    % BREED
    breeders = L(1:parents, :); % make a copy of the top performers
    for j =  1:2:parents % for each pair
        phi1 = ma*rand(1, size(L, 2)) + (.5 - ma/2);
        phi2 = ma*rand(1, size(L, 2)) + (.5 - ma/2);
        L(parents + j, :) = phi1 .*breeders(j, :) + (1 - phi1) .* breeders(j + 1, :);
        L(parents + j + 1, :) = phi2 .*breeders(j, :) + (1 - phi2) .* breeders(j + 1, :);
    end
    
    % REFILL
    L(parents + children + 1:end, :) = (rand(S - parents - children, size(L, 2)) - .5) .* scale + offset;
    
    if mod(g, G/10) == 0 % plot results every few generations and at end
        figure(2); clf; plotDroneResults(PI, MO, LO, TO, parents, g);
        figure(3); clf; familyTree(Orig(1:g, :), parents, parents);
    end
    if mod(g, reset) == 0
        scale = sa*std(L(1:parents, :), 1); offset = mean(L(1:parents, :), 1); 
    end
end % end evolution
 
figure(); % RUNTIMES FOR EACH GENERATION
plot(1:g, sum(comp_time(1:g, :), 2), '-', 'Color', [.8 .2 .6], 'LineWidth', 1);
ylabel("Runtime (s)"); xlabel("Generation"); title("Runtime History");
 
fprintf("Total run time: " + sec2Clock(sum(comp_time(:))) + "\n\n");
 
%% HELPER FUNCTIONS
 
function out = closestFactors(n)
% Returns the two integers that are closest to being square roots of n.
% Useful for making square-ish rectangles.
out = []; i = floor(sqrt(n));
while i >= 1 && isempty(out)
    if mod(n, i) == 0; out = [i, n/i]; end; i = i - 1;
end
end
 
function plotDomain(x, Tx, Ox, g, s, t)
plot3(x(:, 1), x(:, 2), x(:, 3), 'k.'); hold on
plot3(Tx(:, 1), Tx(:, 2), Tx(:, 3), 'go'); 
plot3(Ox(:, 1), Ox(:, 2), Ox(:, 3), 'r*');
xlabel("x (m)"); ylabel("y (m)"); xlabel("z (m)");
title(['g: '  num2str(g)  ', s: ' num2str(s) ', t: ' num2str(t)]);
view(3); axis equal; shg; 
end
 
function plotCost(PI, parents, g)
% a general method for plotting the cost convergence of a series of genetic
% algorithm (or similar) iuterative design options.
semilogy(1:g, PI(1:g, 1), '-', 'Color', [.1 .9 .3], 'LineWidth', 3); hold on;
semilogy(1:g, mean(PI(1:g, 1:parents), 2), '-', 'Color', [1 .7 .2], 'LineWidth', 3); hold on;
semilogy(1:g, mean(PI(1:g, :), 2), '-', 'Color', [1 .2 .2], 'LineWidth', 3);
title("Cost"); xlabel("Generation"); ylabel("\Pi");
legend("Best", "Parent Mean", "Overall Mean", 'location', 'best'); shg
end
 
function plotDroneResults(PI, AO, LO, TO, parents, g)
% produces visualizations specific to the agent assignment
subplot(1, 2, 1) % cost function graph
plotCost(PI, parents, g)
 
subplot(1, 2, 2) % individual performance specs graph
plot(1:g, AO(1:g, 1), "b-", 'LineWidth', 3); hold on;
plot(1:g, mean(AO(1:g, 1:parents),  2), "b-.", 'LineWidth', 2); hold on;
plot(1:g, mean(AO(1:g, :),  2), "b:", 'LineWidth', 2); hold on;
plot(1:g, LO(1:g, 1), "k-", 'LineWidth', 3); hold on;
plot(1:g, mean(LO(1:g, 1:parents),  2), "k-.", 'LineWidth', 2); hold on;
plot(1:g, mean(LO(1:g, :),  2), "k:", 'LineWidth', 2); hold on;
plot(1:g, TO(1:g, 1), "m-", 'LineWidth', 3); hold on;
plot(1:g, mean(TO(1:g, 1:parents), 2), "m-.", 'LineWidth', 2);
plot(1:g, mean(TO(1:g, :), 2), "m:", 'LineWidth', 2);
 
title("Performance"); xlabel("Generation"); ylabel("Fraction of Max");
legend("Task: Best", "Task: Parent Mean", "Task: Overall Mean", "Losses: Best", "Losses: Parent Mean", "Losses: Overall Mean", "Time Use: Best", "Time Use: Parent Mean", "Time Use: Overall Mean",  'location', 'best'); shg
end
 
function familyTree(Orig, parents, pop)
%%
 
% Inputs: 
%   Orig -- the indices of a sorted GA generation from before sorting (see sort() documentation), 
%   parents -- the number of parents, required for interpreting source
%   pop --  the number of performers to plot. Use pop >= parents. 
 
% Returns: 
% no variables, plots a representation of the evolution history to the current figure. 
 
% The function automatically ignores generations where no rank changes
% occur among the parents OR there are any repeated indices (indicating
% incorrect data). 
 
% Data visualization: This function can be used to visualize and interpret the performance of
% your GA iteration process. Gray lines represent "survival" with or
% without a rank change. Red lines represent breeding (e.g.,  a new string
% will be connected to its parents with red lines). New random strings have
% no connections. A surviving string will be represented with gray, a new
% string generated randomly will be a blue mark and a new string generated
% by breeding will be red.
 
% Performance interpretation: If your GA is working correctly, there should
% be lots of turnover (i.e., few generations where nothing happens),
% significant numbers of successful offspring and a moderate number of
% successful random strings. You can also spot stagnation (a parent
% surviving for many generations and continually producing offspring that
% stay in the top ranks). 
 
%%
 
Orig2 = Orig(:, 1:pop); % trim source to just relevant entries.
row = 0; changes = []; % initialize variables for determining relevant generations
children = parents; % assume nearest-neighbor with two children
rando = zeros(0, 2); inc = zeros(0, 2); kid = zeros(0, 2); % intialize empty storage arrays
G = size(Orig, 1); % total number of generations.
pts = 25; % number of points to plot in connections
c1 = [.6 .6 .6]; c2 = [1 .6 .6]; % line colors for surviving connections and children
lw = 1.5; % connection line weight
mw = 1.5; % marker line weight
 
incx = zeros(pts, 2); incy = zeros(pts, 2); % empty arrays for connecting line coordinates.
kidx = zeros(pts, 2); kidy = zeros(pts, 2);
 
for g = 1:G % for every generation
    if ~isequal(Orig2(g, 1:parents), 1:parents) && length(unique(Orig2(g, :))) == pop % if a change in survivors and valid data
        row = row + 1; % row on which to plot current state - counts relevant generations
        x1 = row - 1; x2 = row; % start and end points of connections
        changes = [changes; g]; % record that a change occured in this generation
        for i = 1:pop
            s = Orig2(g, i); y2 = i;
            if s == i && i <= parents && g > 1 % if the entry is a surviving parent who has not moved
                y1 = i;
                [xx, yy] = mySpline([x1 x2], [y1 y2], pts);
                incx = [incx, xx]; incy = [incy, yy];
                inc = [inc; [x2, y2]];
            elseif  s <= parents && g > 1% if the entry is a surviving parent who has been moved down
                y1 = s;
                [xx, yy] = mySpline([x1 x2], [y1 y2], pts);
                incx = [incx, xx]; incy = [incy, yy];
                inc = [inc; [x2, y2]];
            elseif s <= parents + children && g > 1 % if the entry is a child
                for n = 2:2:children
                    if s <= parents + n
                        y11 = n - 1; y12 = n;
                        [xx1, yy1] = mySpline([x1, x2], [y11, y2], pts);
                        [xx2, yy2] = mySpline([x1, x2], [y12, y2], pts);
                        kidx = [kidx, xx1, xx2]; kidy = [kidy, yy1, yy2];
                        kid = [kid; [x2, y2]];
                        break
                    end
                end
            else % if it's a new random addition.
                rando = [rando; [x2, y2]];
            end
        end
    end
end
 
p1 = plot(incx, incy, '-', 'Color', c1, 'LineWidth', 1.5); hold on
p2 = plot(kidx, kidy, '-', 'Color', c2, 'LineWidth', 1.5); hold on
p3 = plot(rando(:, 1), rando(:, 2), 's', 'MarkerEdgeColor', [.2 .4 .9], 'MarkerFaceColor', [.6 .6 1], 'MarkerSize', 10, 'LineWidth', mw); hold on % plot random
p4 = plot(inc(:, 1), inc(:, 2), 's', 'MarkerEdgeColor', [.3 .3 .3], 'MarkerFaceColor', [.6 .6 .6], 'MarkerSize', 10, 'LineWidth', mw); hold on % plot survival
p5 = plot(kid(:, 1), kid(:, 2), 's', 'MarkerEdgeColor', [.9 .3 .3], 'MarkerFaceColor', [1 .6 .6], 'MarkerSize', 10, 'LineWidth', mw); % plot children
h = [p3, p4, p5];
legend(h, "Random", "Incumbent", "Child")
 
xlabels = {};
for i = 1:numel(changes)
    xlabels{i} = num2str(changes(i));
end
 
ylabels = {};
for j = 1:pop
    ylabels{j} = num2str(j);
end
 
title("Family Tree");
set(gca, 'xtick', [1:row]); set(gca,'ytick', [1:pop]);
set(gca,'xticklabel', xlabels); set(gca,'yticklabel', ylabels);
xlabel("generation"); ylabel("Rank");
axis([0 row + 1 0 pop + 1]); view([90, 90])
end
 
function [xx, yy] = mySpline(x, y, pts)
% produces clamped splines between two points 
 
% Inputs: 
%   x -- 2-value vector containing the x coordinates of the end points
%   y -- 2-value vector containing the y coordinates of the end points
%   pts -- the number of total points to plot (ends plus intermediates)
 
% Returns: 
%    xx -- array of x coordinates to plot
%    yy -- array of y coordinates to plot
 
cs = spline(x, [0 y 0]);
xx = linspace(x(1), x(2), pts)';
yy = ppval(cs,xx);
end
 
function myProgressBar(tock, t, tf, m)
% Provides a status update on a calculation, printed to the command window.
% avoids the issues associated with the GUI-based equilvalent functions. 
% 
% Inputs: 
%   tock -- total run time so far, seconds
%   t -- cycles, simulation time, etc completed
%   tf -- total cycles, etc required for process
%   m -- optional message input, empty by default
 
% Outputs: 
% no variables, prints information to the command window in the format: 
% "#######-----------" + "m" + "Time remaining: " + "HH:MM:SS"
 
if nargin < 4
    m = '';
end
rem = max(tock*(tf/t - 1), 0);
clock = sec2Clock(rem);
totchar = 20;
fprintf(repeatStr("#", floor(t/tf*totchar)) + repeatStr("-", totchar - floor(t/tf*totchar)) ...
    + "   " + m + "   Time remaining: " + clock + "\n\n");
end
 
function out = sec2Clock(s)
% returns a string of characters that represents a number of seconds in
% format HH:MM:SS. Can include arbitrarily many hour digits to account for
% large times. Rounded down to nearest number of seconds. 
    remh = floor(s / 3600); s = s - 3600*remh; remm = floor(s / 60); s = s - 60*remm; rems = floor(s); 
    out = padStr(num2str(remh), "0", 2)  + ":" + padStr(num2str(remm), "0", 2) ...
    + ":" + padStr(num2str(rems), "0", 2); 
end
 
function out = padStr(st, pad, minlength)
% returns string st plus some number of pad characters such that the total
% string length is as least minlength. Puts the padding on the left side. 
out = st; 
while (strlength(out) < minlength) out = pad + out; end
end
 
function out = repeatStr(st, n)
% returns a given string st repeated n times
out = ""; for i = 1:n out = out + st; end
end

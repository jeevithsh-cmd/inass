function acf_xai_fl(section, outputs_dir)
% ACF_XAI_FL  MATLAB analysis of ACF-XAI-FL training outputs.
%
% All numbers are READ from the files written by scripts/main_train.py;
% nothing is simulated or re-digitised.
%
%   acf_xai_fl                 % run all sections on ./outputs
%   acf_xai_fl('conv')         % Section 1: convergence curves (Figure 2, Table 6)
%   acf_xai_fl('tsmc')         % Section 2: TSMC aggregation weights per round
%   acf_xai_fl('stats')        % Section 3: paired Wilcoxon signed-rank tests
%   acf_xai_fl('noise')        % Section 4: WT DSC vs. sigma (Table 8 / sensitivity figure)
%   acf_xai_fl('all', 'path/to/outputs')
%
% Expected layout: <outputs>/<experiment>/seed_<s>/{history.csv,test_per_patient.csv,test_summary.json}
% Experiment names used below follow README.md ("Reproducing every table").
% Requires MATLAB R2020a+; 'signrank' needs the Statistics and Machine Learning Toolbox.

if nargin < 1 || isempty(section), section = 'all'; end
if nargin < 2, outputs_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'outputs'); end
fprintf('ACF-XAI-FL analysis | outputs = %s\n\n', outputs_dir);

switch lower(section)
    case 'conv',  section_convergence(outputs_dir);
    case 'tsmc',  section_tsmc_weights(outputs_dir);
    case 'stats', section_wilcoxon(outputs_dir);
    case 'noise', section_noise(outputs_dir);
    case 'all'
        section_convergence(outputs_dir);
        section_tsmc_weights(outputs_dir);
        section_wilcoxon(outputs_dir);
        section_noise(outputs_dir);
    otherwise, error('Unknown section "%s"', section);
end
end

% =========================================================================
function [r, wt] = read_curve(seed_dir)
T = readtable(fullfile(seed_dir, 'history.csv'), 'Delimiter', ',', 'TextType', 'string');
v = T.val_WT_DSC;
if ~isnumeric(v), v = str2double(v); end
ok = ~isnan(v);
r = T.round(ok); wt = v(ok);
end

function S = read_summary(seed_dir)
S = jsondecode(fileread(fullfile(seed_dir, 'test_summary.json')));
end

function d = seed_dirs(exp_dir)
L = dir(fullfile(exp_dir, 'seed_*'));
d = arrayfun(@(x) fullfile(x.folder, x.name), L, 'UniformOutput', false);
end

% =========================================================================
% SECTION 1 — Convergence (Figure 2, Table 6)
% =========================================================================
function section_convergence(od)
fprintf('--- Section 1: convergence (validation WT DSC vs. round) ---\n');
exps = {'acf_xai_fl_full', 'fedavg'};
cen = fullfile(od, 'centralised');
target = NaN;
if isfolder(cen)
    s = cellfun(@read_summary, seed_dirs(cen));
    target = 0.85 * mean(arrayfun(@(x) x.segmentation_and_xai.WT_DSC, s));
    fprintf('  85%% of centralised test WT DSC = %.4f\n', target);
end
figure('Name', 'Convergence'); hold on; grid on;
styles = {'b-', 'r--'};
for e = 1:numel(exps)
    ed = fullfile(od, exps{e});
    if ~isfolder(ed), fprintf('  [skip] %s not found\n', exps{e}); continue; end
    D = seed_dirs(ed); curves = []; hits = nan(numel(D), 1);
    for i = 1:numel(D)
        [r, wt] = read_curve(D{i});
        curves(:, i) = wt; %#ok<AGROW>
        k = find(wt >= target, 1);
        if ~isempty(k), hits(i) = r(k); end
    end
    m = mean(curves, 2); sd = std(curves, 0, 2);
    fill([r; flipud(r)], [m - sd; flipud(m + sd)], styles{e}(1), 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
         'HandleVisibility', 'off');
    plot(r, m, styles{e}, 'LineWidth', 2, 'DisplayName', exps{e});
    fprintf('  %-16s final val WT = %.4f +- %.4f | rounds to target = %s\n', exps{e}, m(end), sd(end), ...
            mat2str(hits'));
end
if ~isnan(target), yline(target, 'k:', '85% centralised'); end
xlabel('Communication round'); ylabel('Validation WT DSC'); legend('Location', 'southeast');
saveas(gcf, fullfile(od, 'convergence_curve.png'));
fprintf('  saved convergence_curve.png\n\n');
end

% =========================================================================
% SECTION 2 — TSMC aggregation weights logged during training
% =========================================================================
function section_tsmc_weights(od)
fprintf('--- Section 2: TSMC reliability-aware aggregation weights ---\n');
D = seed_dirs(fullfile(od, 'acf_xai_fl_full'));
if isempty(D), fprintf('  [skip] acf_xai_fl_full not found\n\n'); return; end
T = readtable(fullfile(D{1}, 'history.csv'), 'Delimiter', ',', 'TextType', 'string');
W = cell2mat(arrayfun(@(s) str2double(split(s, ';'))', T.tsmc_weights, 'UniformOutput', false));
figure('Name', 'TSMC weights'); plot(T.round, W, 'LineWidth', 1.5); grid on;
legend({'A (T1ce)', 'B (T2)', 'C (FLAIR)', 'D (multi)'}, 'Location', 'best');
xlabel('Communication round'); ylabel('Aggregation weight w_i^t');
saveas(gcf, fullfile(od, 'tsmc_weights.png'));
fprintf('  final-round weights (%s): %s\n\n', D{1}, mat2str(W(end, :), 4));
end

% =========================================================================
% SECTION 3 — Paired Wilcoxon signed-rank tests on per-patient WT DSC
% =========================================================================
function section_wilcoxon(od)
fprintf('--- Section 3: Wilcoxon signed-rank (per-patient WT DSC, seeds averaged) ---\n');
ref = 'acf_xai_fl_full';
R = per_patient(fullfile(od, ref));
if isempty(R), fprintf('  [skip] %s not found\n\n', ref); return; end
L = dir(od); L = L([L.isdir] & ~startsWith({L.name}, '.'));
fprintf('  %-22s | n   | ref mean | cmp mean | p-value\n', 'compared with');
for i = 1:numel(L)
    if strcmp(L(i).name, ref) || ~isfolder(fullfile(od, L(i).name, 'seed_42')), continue; end
    C = per_patient(fullfile(od, L(i).name));
    [ids, ia, ib] = intersect(R.pid, C.pid);
    if numel(ids) < 5, continue; end
    p = signrank(R.wt(ia), C.wt(ib));
    fprintf('  %-22s | %3d | %.4f   | %.4f   | %.3e\n', L(i).name, numel(ids), mean(R.wt(ia)), mean(C.wt(ib)), p);
end
fprintf('\n');
end

function P = per_patient(exp_dir)
P = [];
D = seed_dirs(exp_dir);
if isempty(D), return; end
allT = [];
for i = 1:numel(D)
    f = fullfile(D{i}, 'test_per_patient.csv');
    if isfile(f), allT = [allT; readtable(f, 'TextType', 'string')]; end %#ok<AGROW>
end
G = groupsummary(allT, 'pid', 'mean', 'WT_DSC');
P.pid = G.pid; P.wt = G.mean_WT_DSC;
end

% =========================================================================
% SECTION 4 — Sensitivity to the round-level perturbation sigma (Table 8)
% =========================================================================
function section_noise(od)
fprintf('--- Section 4: WT DSC vs. sigma (round-level Gaussian perturbation) ---\n');
sig = [0, 0.01, 0.05, 0.10, 0.20];
names = {'sigma_0', 'acf_xai_fl_full', 'sigma_0.05', 'sigma_0.10', 'sigma_0.20'};
m = nan(size(sig)); s = m;
for i = 1:numel(sig)
    D = seed_dirs(fullfile(od, names{i}));
    if isempty(D), continue; end
    v = cellfun(@(d) read_summary(d).segmentation_and_xai.WT_DSC, D);
    m(i) = mean(v); s(i) = std(v);
    fprintf('  sigma = %.2f  WT DSC = %.4f +- %.4f  (%s)\n', sig(i), m(i), s(i), names{i});
end
if all(isnan(m)), fprintf('  [skip] no sigma runs found\n\n'); return; end
figure('Name', 'Noise sensitivity'); errorbar(sig, m, s, 'o-', 'LineWidth', 1.5); grid on;
xline(0.01, '--', 'default \sigma = 0.01');
xlabel('\sigma (noise std, round-level; no (\epsilon,\delta) accounting)'); ylabel('Test WT DSC');
saveas(gcf, fullfile(od, 'noise_sensitivity.png'));
fprintf('  saved noise_sensitivity.png\n\n');
end

data_path = '/data/challenge-2015/data/';
addpath(genpath('/home/alistairewj/git/peak-detector'));
addpath('/data/challenge-2015/wfdb-app-toolbox-0-9-9/mcode/');
fp = fopen([data_path 'ALARMS'],'r');
alarms=textscan(fp,'%s%s%d','delimiter',',');
fclose(fp);
records=alarms{1};
targets=alarms{3};
alarms=alarms{2};


% define input options for the peak detector
% all of the options listed here are the default values, and are optionally omitted
opt = struct(...
    'SIZE_WIND',10,... % define the window for the bSQI check on the ECG
    'LG_MED',3,... % take the median SQI using X nearby values,  so if LG_MED = 3, we take the median of the 3 prior and 3 posterior windows
    'REG_WIN',1,... % how frequently to check the SQI for switching - i.e., if REG_WIN = 1, then we check the signals every second to switch
    'THR',0.150,... % the width, in seconds, used when comparing peaks in the F1 based ECG SQI
    'SQI_THR',0.8,... % the SQI threshold - we switch signals if SQI < this value
    'USE_PACING',1,... % flag turning on/off the pacing detection/correction
    'ABPMethod','wabp',... % ABP peak detection method (wabp, delineator)
    'SIMPLEMODE', 1,... % simple mode only uses the first ABP and ECG signal, and ignores all others
    'DELAYALG', 'map',... % algorithm used to determine the delay between the ABP and the ECG signal
    'SAVE_STUFF', 0,... % leave temporary files in working directory
    ... % jqrs parameters - the custom peak detector implemented herein
    'JQRS_THRESH', 0.3,... % energy threshold for defining peaks
    'JQRS_REFRAC', 0.25,... % refractory period in seconds
    'JQRS_INTWIN_SZ', 7,...
    'JQRS_WINDOW', 15);

% copy file to this folder
for i = 1:numel(records)
    recordName = records{i};
    
   % create symlinks for data
    system(['ln -frs ' data_path recordName '.mat ' recordName '.mat']);
    system(['ln -frs ' data_path recordName '.hea ' recordName '.hea']);
    
    % run peak detector
    [t,data] = rdsamp(recordName);
    [siginfo,fs] = wfdbdesc(recordName);
    
    % extract info from structure output by wfdbdesc
    header = arrayfun(@(x) x.Description, siginfo, 'UniformOutput', false);
    
    % run SQI based switching
    %[ qrs, sqi, qrs_comp, qrs_header ] = detect_sqi(data, header, fs, opt);
    
    
    
    [ idxECG, idxABP, idxPPG, idxSV ] = getSignalIndices(header);
    idxECG = idxECG(:)';
    if ~isempty(idxECG)
        for m = idxECG
            opt.LG_REC = size(data,1) ./ fs(m); % length of the record in seconds
            opt.N_WIN = ceil(opt.LG_REC/opt.REG_WIN); % number of windows in the signal
            ann_jqrs = run_qrsdet_by_seg_ali(data(:,m),fs(m),opt);
            if isempty(ann_jqrs)
                fprintf('%s - empty signal.\n',recordName);
                system('rm tmp; touch tmp');
                % make an empty annotation file for jqrs
                system(['wrann -r ' recordName ' -a jqrs <tmp']);
                system('rm tmp');
            else
                wrann(recordName,'jqrs',ann_jqrs)
                fprintf('%s %5s - %d QRS peaks.\n',recordName,header{m},numel(ann_jqrs));
            end
            system(['mv ' recordName '.jqrs /data/challenge-2015/ann/' recordName '.jqrs' num2str(m-1) ]);
        end
    else
        fprintf('%s - no ECG.\n',recordName);
    end
    system(['rm ' recordName '.mat']);
    system(['rm ' recordName '.hea']);
end
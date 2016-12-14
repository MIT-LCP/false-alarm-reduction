
# coding: utf-8

# # Overall pipeline

# In[2]:

from datetime                      import datetime
import invalid_sample_detection    as invalid
import evaluation                  as evaluate
import load_annotations            as annotate
import regular_activity            as regular
import specific_arrhythmias        as arrhythmia
import numpy                       as np
import parameters
import os
import csv
import wfdb

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
ecg_ann_type = 'gqrs'


# ## Classifying arrhythmia alarms

# In[3]:

# Returns true if alarm is classified as a true alarm
def is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type, verbose=False): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, is_true_alarm = regular.check_gold_standard_classification(fields)

    is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type, alarm_type, should_check_nan=False)    
    if is_regular:
        if verbose: 
            print sample_name + "with regular activity"
        return False
    
    if alarm_type == "Asystole": 
        arrhythmia_test = arrhythmia.test_asystole
    elif alarm_type == "Bradycardia": 
        arrhythmia_test = arrhythmia.test_bradycardia
    elif alarm_type == "Tachycardia": 
        arrhythmia_test = arrhythmia.test_tachycardia
    elif alarm_type == "Ventricular_Tachycardia": 
        arrhythmia_test = arrhythmia.test_ventricular_tachycardia
    elif alarm_type == "Ventricular_Flutter_Fib": 
        arrhythmia_test = arrhythmia.test_ventricular_flutter_fibrillation
    else: 
        raise Exception("Unknown arrhythmia alarm type")
    
    try: 
        classified_true_alarm = arrhythmia_test(data_path, ann_path, sample_name, ecg_ann_type, verbose)
        return classified_true_alarm

    except Exception as e: 
        print "sample_name: ", sample_name, e


def is_true_alarm(data_path, sample_name): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, true_alarm = regular.check_gold_standard_classification(fields)
    return true_alarm


# In[4]:

# Generate confusion matrix for all samples given sample name/directory
def generate_confusion_matrix_dir(data_path, ann_path, ecg_ann_type): 
    confusion_matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }
    
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)
            
            true_alarm = is_true_alarm(data_path, sample_name)
            classified_true_alarm = is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type)

            matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
            confusion_matrix[matrix_classification].append(sample_name)
            if matrix_classification == "FN": 
                print "FALSE NEGATIVE: ", filename
                
    return confusion_matrix


def get_confusion_matrix_classification(true_alarm, classified_true_alarm): 
    if true_alarm and classified_true_alarm: 
        matrix_classification = "TP"

    elif true_alarm and not classified_true_alarm: 
        matrix_classification = "FN"

    elif not true_alarm and classified_true_alarm: 
        matrix_classification = "FP"

    else: 
        matrix_classification = "TN"

    return matrix_classification


# In[5]:

def print_by_type(false_negatives): 
    counts_by_type = {}
    for false_negative in false_negatives: 
        first = false_negative[0] 
        if first not in counts_by_type.keys(): 
            counts_by_type[first] = 0
        counts_by_type[first] += 1

    print counts_by_type
    
    
def print_by_arrhythmia(confusion_matrix, arrhythmia_prefix): 
    counts_by_arrhythmia = {}
    for classification_type in confusion_matrix.keys(): 
        sample_list = [ sample for sample in confusion_matrix[classification_type] if sample[0] == arrhythmia_prefix]
        counts_by_arrhythmia[classification_type] = (len(sample_list), sample_list)

    print counts_by_arrhythmia
    
def get_counts(confusion_matrix): 
    return { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }


# In[ ]:

if __name__ == '__main__': 
    start = datetime.now() 
    confusion_matrix_gqrs = generate_confusion_matrix_dir(data_path, ann_path, 'gqrs')
    counts_gqrs = get_counts(confusion_matrix_gqrs)
    print "confusion matrix: ", confusion_matrix_gqrs
    print "total time: ", datetime.now() - start

    evaluate.print_stats(counts_gqrs)
    print_by_type(confusion_matrix_gqrs['FN'])
    print_by_arrhythmia(confusion_matrix_gqrs, 'v')
    
    fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
    print "missed true positives: ", get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TP")
    print "missed true negatives: ", get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TN")


# Regular algorithm: 
# 
# counts:  {'FP': 95, 'TN': 361, 'TP': 249, 'FN': 45}
# sensitivity:  0.84693877551
# specificity:  0.791666666667
# ppv:  0.723837209302
# f1:  0.780564263323
# score:  0.655913978495
# {'a': 1, 'b': 8, 't': 2, 'v': 34}
# 
# print by vtach: 
# {'FP': (78, ['v718s', 'v619l', 'v141l', 'v322s', 'v361l', 'v826s', 'v470s', 'v611l', 'v475l', 'v540s', 'v364s', 'v365l', 'v243l', 'v147l', 'v373l', 'v763l', 'v518s', 'v232s', 'v182s', 'v122s', 'v686s', 'v846s', 'v337l', 'v233l', 'v640s', 'v136s', 'v713l', 'v552s', 'v244s', 'v502s', 'v113l', 'v531l', 'v781l', 'v513l', 'v241l', 'v551l', 'v347l', 'v804s', 'v791l', 'v198s', 'v473l', 'v479l', 'v687l', 'v250s', 'v590s', 'v814s', 'v463l', 'v535l', 'v843l', 'v767l', 'v609l', 'v119l', 'v759l', 'v375l', 'v390s', 'v296s', 'v205l', 'v295l', 'v222s', 'v472s', 'v834s', 'v570s', 'v489l', 'v492s', 'v115l', 'v483l', 'v721l', 'v204s', 'v323l', 'v498s', 'v775l', 'v312s', 'v830s', 'v181l', 'v621l', 'v405l', 'v682s', 'v481l']), 'TN': (174, ['v135l', 'v676s', 'v283l', 'v201l', 'v177l', 'v168s', 'v294s', 'v620s', 'v200s', 'v291l', 'v663l', 'v666s', 'v392s', 'v658s', 'v482s', 'v711l', 'v627l', 'v317l', 'v634s', 'v395l', 'v370s', 'v647l', 'v259l', 'v419l', 'v374s', 'v148s', 'v581l', 'v166s', 'v224s', 'v519l', 'v459l', 'v271l', 'v380s', 'v655l', 'v246s', 'v324s', 'v848s', 'v671l', 'v431l', 'v258s', 'v808s', 'v536s', 'v623l', 'v179l', 'v644s', 'v566s', 'v643l', 'v437l', 'v169l', 'v770s', 'v511l', 'v533l', 'v501l', 'v367l', 'v280s', 'v532s', 'v732s', 'v845l', 'v319l', 'v140s', 'v720s', 'v282s', 'v401l', 'v217l', 'v438s', 'v336s', 'v811l', 'v649l', 'v176s', 'v466s', 'v127l', 'v505l', 'v766s', 'v704s', 'v355l', 'v274s', 'v211l', 'v202s', 'v480s', 'v146s', 'v692s', 'v585l', 'v612s', 'v660s', 'v102s', 'v257l', 'v809l', 'v727l', 'v633l', 'v381l', 'v491l', 'v242s', 'v210s', 'v774s', 'v738s', 'v326s', 'v316s', 'v403l', 'v548s', 'v433l', 'v360s', 'v421l', 'v399l', 'v247l', 'v292s', 'v743l', 'v423l', 'v111l', 'v307l', 'v756s', 'v155l', 'v398s', 'v218s', 'v354s', 'v426s', 'v476s', 'v453l', 'v153l', 'v154s', 'v795l', 'v289l', 'v402s', 'v128s', 'v575l', 'v262s', 'v338s', 'v162s', 'v779l', 'v245l', 'v749l', 'v180s', 'v598s', 'v784s', 'v782s', 'v615l', 'v557l', 'v725l', 'v303l', 'v298s', 'v674s', 'v256s', 'v601l', 'v460s', 'v464s', 'v736s', 'v833l', 'v212s', 'v827l', 'v100s', 'v160s', 'v583l', 'v366s', 'v325l', 'v293l', 'v327l', 'v710s', 'v549l', 'v452s', 'v432s', 'v427l', 'v568s', 'v400s', 'v207l', 'v101l', 'v569l', 'v248s', 'v371l', 'v510s', 'v230s', 'v641l', 'v164s', 'v359l', 'v454s', 'v353l']), 'TP': (55, ['v221l', 'v783l', 'v788s', 'v368s', 'v758s', 'v329l', 'v652s', 'v255l', 'v729l', 'v541l', 'v769l', 'v823l', 'v616s', 'v632s', 'v542s', 'v803l', 'v772s', 'v842s', 'v522s', 'v836s', 'v844s', 'v748s', 'v275l', 'v646s', 'v815l', 'v559l', 'v309l', 'v404s', 'v625l', 'v828s', 'v254s', 'v318s', 'v328s', 'v143l', 'v837l', 'v596s', 'v199l', 'v635l', 'v253l', 'v726s', 'v573l', 'v714s', 'v564s', 'v159l', 'v369l', 'v733l', 'v194s', 'v648s', 'v638s', 'v765l', 'v628s', 'v630s', 'v188s', 'v132s', 'v290s']), 'FN': (34, ['v206s', 'v523l', 'v334s', 'v448s', 'v579l', 'v158s', 'v813l', 'v831l', 'v597l', 'v797l', 'v571l', 'v724s', 'v197l', 'v773l', 'v696s', 'v525l', 'v701l', 'v471l', 'v728s', 'v636s', 'v574s', 'v131l', 'v805l', 'v139l', 'v761l', 'v348s', 'v806s', 'v793l', 'v133l', 'v629l', 'v626s', 'v818s', 'v534s', 'v607l'])}
# 
# 
# confusion matrix: 
# {'FP': ['v718s', 'v619l', 'v141l', 'v322s', 'v361l', 't504s', 'v826s', 'v470s', 'v611l', 'v475l', 'v540s', 'v364s', 'b389l', 'v365l', 'v243l', 'a778s', 'v147l', 'v373l', 'a306s', 'v763l', 'v518s', 'v232s', 'v182s', 'v122s', 'v686s', 'v846s', 'v337l', 'v233l', 'a645l', 'v640s', 'v136s', 'v713l', 'v552s', 'v244s', 'a123l', 'v502s', 'b487l', 'a219l', 'v113l', 'v531l', 'v781l', 'v513l', 'a825l', 'v241l', 'a539l', 'v551l', 'v347l', 'v804s', 'v791l', 'v198s', 'v473l', 'v479l', 'v687l', 'v250s', 'v590s', 'v814s', 'v463l', 'v535l', 'v843l', 'v767l', 'v609l', 'v119l', 'a279l', 'v759l', 'v375l', 'f529l', 'v390s', 'v296s', 'v205l', 'v295l', 'v222s', 'a699l', 'v472s', 't409l', 'v834s', 'v570s', 'v489l', 'v492s', 'a668s', 'a462s', 'v115l', 'v483l', 'v721l', 'v204s', 'v323l', 'v498s', 'v775l', 'v312s', 'v830s', 'a391l', 'v181l', 'v621l', 'v405l', 'v682s', 'v481l'], 'TN': ['v135l', 'a226s', 'f593l', 'a582s', 'v676s', 'v283l', 'a735l', 'v201l', 'b484s', 'a267l', 'a239l', 'f415l', 'v177l', 'v168s', 'f586s', 't383l', 'a378s', 'a376s', 'v294s', 'v620s', 'f792s', 'f189l', 'v200s', 'f407l', 'v291l', 't469l', 'v663l', 'a715l', 'v666s', 'a673l', 'f236s', 'v392s', 'a591l', 'v658s', 'a363l', 'a785l', 'b231l', 'v482s', 'v711l', 'f304s', 'b451l', 'v627l', 'v317l', 'f789l', 'a377l', 'v634s', 'a272s', 'a163l', 'a152s', 'f237l', 'v395l', 'a750s', 'v370s', 'v647l', 'v259l', 'v419l', 'a514s', 'a819l', 'v374s', 'b753l', 'v148s', 'f613l', 'f138s', 'v581l', 'f321l', 'b849l', 'b330s', 'b703l', 'v166s', 'b835l', 'v224s', 'v519l', 'v459l', 'v271l', 'v380s', 'a555l', 'v655l', 'v246s', 'b617l', 'v324s', 'f610s', 'v848s', 'v671l', 'b308s', 'v431l', 'v258s', 'a105l', 'a396s', 'v808s', 'a798s', 'v536s', 'v623l', 'v179l', 'b428s', 'b285l', 'v644s', 'v566s', 'f530s', 'f144s', 'b486s', 'v643l', 'v437l', 'f130s', 'a606s', 'v169l', 'b681l', 'v770s', 'v511l', 'a435l', 'a461l', 'a165l', 'v533l', 'a780s', 'a807l', 'a512s', 'a297l', 'a311l', 'b332s', 'v501l', 'v367l', 'b554s', 'f751l', 'v280s', 'f642s', 'v532s', 'f362s', 'v732s', 'a527l', 'b528s', 'v845l', 'f137l', 'v319l', 'v140s', 'v720s', 'b485l', 'f352s', 'a134s', 'a310s', 'v282s', 'b340s', 'f261l', 'b553l', 'a287l', 'a624s', 'v401l', 'v217l', 'v438s', 'a650s', 'a847l', 'a490s', 'v336s', 'a631l', 'b184s', 'v811l', 'v649l', 'v176s', 'v466s', 'v127l', 'a651l', 'v505l', 'v766s', 'v704s', 'a661l', 'b314s', 'v355l', 'f196s', 'v274s', 'a429l', 'v211l', 'f493l', 'f346s', 'b587l', 'v202s', 'v480s', 'v146s', 'v692s', 'v585l', 'a178s', 'v612s', 'b388s', 'f120s', 'a109l', 'v660s', 'a667l', 'v102s', 'v257l', 'v809l', 'v727l', 'b824s', 'v633l', 'a315l', 'a302s', 'f691l', 'b339l', 'v381l', 'a301l', 'v491l', 'v242s', 'a603l', 'f408s', 'v210s', 'v774s', 'f500s', 'b669l', 'v738s', 'v326s', 'f572s', 'v316s', 'v403l', 'v548s', 'b349l', 'a225l', 'v433l', 'b331l', 'v360s', 'b600s', 'v421l', 'a712s', 'a104s', 'v399l', 'v247l', 'v292s', 'v743l', 'b841l', 'v423l', 'v111l', 'v307l', 'a556s', 'v756s', 'a397l', 'v155l', 'b216s', 'f260s', 'f474s', 't503l', 'a599l', 't496s', 'a266s', 'v398s', 'f592s', 'a746s', 'b685l', 'v218s', 'a288s', 'f799l', 'a802s', 'b706s', 'v354s', 'v426s', 'a675l', 'v476s', 'v453l', 'a278s', 'f576s', 'v153l', 'f602s', 'f637l', 'v154s', 'v795l', 'v289l', 'v402s', 'v128s', 'v575l', 'v262s', 'v338s', 'v162s', 't116s', 'a822s', 'f829l', 'v779l', 'a723l', 'v245l', 'v749l', 'v180s', 'v598s', 'a420s', 'a457l', 'v784s', 'v782s', 't817l', 'f129l', 'a439l', 'a382s', 'b488s', 'v615l', 'a145l', 'b341l', 'f657l', 'a558s', 'v557l', 'a694s', 'v725l', 'b387l', 'v303l', 'f441l', 'v298s', 'v674s', 'v256s', 'a740s', 'a273l', 'a608s', 'v601l', 'a526s', 'v460s', 'a465l', 'v464s', 'v736s', 'f414s', 'v833l', 'v212s', 'v827l', 'v100s', 'f440s', 'v160s', 'v583l', 'v366s', 'v325l', 'v293l', 'a223l', 'a170s', 'v327l', 'a186s', 'a171l', 'v710s', 'v549l', 'f768s', 'a705l', 'v452s', 'a550s', 'v432s', 'f618s', 'a422s', 'f121l', 'b215l', 'f605l', 'v427l', 'v568s', 'f499l', 'v400s', 'v207l', 'v101l', 'f281l', 'a584s', 'f190s', 'v569l', 'v248s', 'b684s', 'v371l', 'a810s', 'a436s', 'a103l', 'v510s', 'v230s', 'v641l', 'b695l', 't384s', 'b286s', 'v164s', 'v359l', 'v454s', 'v353l'], 'FN': ['v206s', 'v523l', 'v334s', 'v448s', 'b734s', 'a670s', 'v579l', 'b379l', 'v158s', 'v813l', 'v831l', 'v597l', 'v797l', 'v571l', 'v724s', 'v197l', 'b183l', 'b494s', 'v773l', 'v696s', 'v525l', 'v701l', 't700s', 'v471l', 'v728s', 'v636s', 'v574s', 'v131l', 'v805l', 'v139l', 'v761l', 't418s', 'b497l', 'v348s', 'v806s', 'v793l', 'v133l', 'v629l', 'v626s', 'b187l', 'b672s', 'v818s', 'v534s', 'v607l', 'b495l'], 'TP': ['v221l', 'f697l', 't344s', 'v783l', 'v788s', 't393l', 't698s', 'v368s', 'b220s', 'v758s', 't240s', 't520s', 't108s', 'v329l', 'b268s', 'b659l', 'v652s', 'b125l', 't677l', 'v255l', 'v729l', 't467l', 'v541l', 'v769l', 't406s', 't151l', 't208s', 't744s', 'v823l', 'a203l', 't193l', 't547l', 'v616s', 't622s', 't717l', 'v632s', 't702s', 't790s', 'b562s', 't801l', 'b757l', 'b794s', 't343l', 'b838s', 't276s', 't665l', 'b820s', 'b124s', 't445l', 'a172s', 't277l', 'b764s', 'v542s', 'b455l', 'a653l', 'b561l', 't300s', 'b656s', 'v803l', 't565l', 'v772s', 't173l', 'a796s', 't251l', 't394s', 't693l', 't417l', 't580s', 'v842s', 'f543l', 't430s', 'v522s', 't412s', 't333l', 't741l', 'b578s', 't567l', 't270s', 'v836s', 't719l', 't191l', 't508s', 'v844s', 'a446s', 'b269l', 'a654s', 't351l', 'b560s', 't411l', 't688s', 't678s', 'v748s', 't771l', 'b840s', 'v275l', 't690s', 't434s', 'b588s', 'b537l', 't777l', 'v646s', 'a776s', 't707l', 't478s', 'a449l', 'b229l', 't737l', 'v815l', 't335l', 't114s', 'v559l', 'v309l', 'a167l', 't234s', 'b722s', 'v404s', 't468s', 't305l', 't716s', 't521l', 't110s', 't112s', 't762s', 'b228s', 't235l', 'a345l', 't350s', 'v625l', 't760s', 'f450s', 'f544s', 't410s', 't679l', 't425l', 'v828s', 't150s', 'b730s', 'a185l', 'v254s', 't320s', 't117l', 'b516s', 'b227l', 'v318s', 't742s', 't209l', 'a442s', 'v328s', 't213l', 't284s', 't356s', 'b664s', 'v143l', 'v837l', 't689l', 't107l', 'v596s', 't747l', 'v199l', 'b538s', 'a142s', 'v635l', 't787l', 't745l', 'v253l', 'v726s', 't157l', 'v573l', 'f563l', 't507l', 'b313l', 'a385l', 't683l', 't413l', 'v714s', 'v564s', 't709l', 'b299l', 't680s', 't594s', 't509l', 't614s', 'v159l', 't424s', 't416s', 'v369l', 'v733l', 't662s', 't252s', 't249l', 't821l', 'a754s', 'v194s', 'a639l', 't524s', 't214s', 'v648s', 't506s', 't577l', 't812s', 'v638s', 'f545l', 't739l', 'b515l', 'v765l', 't477l', 'a604s', 't175l', 'a386s', 'b708s', 'a372s', 't195l', 'a443l', 't444s', 't358s', 't264s', 'b839l', 'b832s', 't752s', 't174s', 't458s', 't595l', 't589l', 'b265l', 't238s', 't800s', 'v628s', 'v630s', 't118s', 't546s', 'b126s', 'v188s', 't192s', 't447l', 'v132s', 'b456s', 'v290s', 't156s', 't149l', 't342s', 't816s', 'b517l', 't263l', 't786s', 't106s', 't357l', 'a161l', 't755l', 't731l']}

# ## Comparing classification with other algorithms

# In[6]:

def generate_others_confusion_matrices(filename, data_path): 
    others_confusion_matrices = {}
    
    with open(filename, "r") as f: 
        reader = csv.DictReader(f)
        authors = reader.fieldnames[1:]
        for author in authors: 
            others_confusion_matrices[author] = { "TP": [], "FP": [], "FN": [], "TN": [] }
            
        for line in reader: 
            sample_name = line['record name']
            true_alarm = is_true_alarm(data_path, sample_name)
            
            for author in authors: 
                classified_true_alarm = line[author] == '1'
                matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
                
                others_confusion_matrices[author][matrix_classification].append(sample_name)
    
    return others_confusion_matrices
                
    
filename = "sample_data/answers.csv"
others_confusion_matrices = generate_others_confusion_matrices(filename, data_path)    


# In[7]:

for author in others_confusion_matrices.keys(): 
    other_confusion_matrix = others_confusion_matrices[author]
    print author
    counts = get_counts(other_confusion_matrix)
    evaluate.print_stats(counts)
    print_by_type(other_confusion_matrix['FN'])


# In[13]:

def get_missed(confusion_matrix, other_confusion_matrix, classification): 
    missed = []
    
    for sample in other_confusion_matrix[classification]: 
        if sample not in confusion_matrix[classification]: 
            missed.append(sample)
            
    return missed
    
fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
print "missed true positives: ", get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TP")
print "missed true negatives: ", get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TN")


# In[ ]:




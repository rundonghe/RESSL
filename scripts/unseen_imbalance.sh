nohup python -u /evaluation_unseen_imbalance.py --algo PseudoLabel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:4 > /log/log-unseen-imbalance/PseudoLabel_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PseudoLabel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:4 > /log/log-unseen-imbalance/PseudoLabel_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PseudoLabel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:4 > /log/log-unseen-imbalance/PseudoLabel_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PseudoLabel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:4 > /log/log-unseen-imbalance/PseudoLabel_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PseudoLabel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:4 > /log/log-unseen-imbalance/PseudoLabel_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo PiModel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:4 > /log/log-unseen-imbalance/PiModel_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PiModel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:4 > /log/log-unseen-imbalance/PiModel_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PiModel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:4 > /log/log-unseen-imbalance/PiModel_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PiModel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:4 > /log/log-unseen-imbalance/PiModel_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo PiModel --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:4 > /log/log-unseen-imbalance/PiModel_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo UASD --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:5 > /log/log-unseen-imbalance/UASD_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UASD --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:5 > /log/log-unseen-imbalance/UASD_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UASD --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:5 > /log/log-unseen-imbalance/UASD_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UASD --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:5 > /log/log-unseen-imbalance/UASD_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UASD --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:5 > /log/log-unseen-imbalance/UASD_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo FixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:5 > /log/log-unseen-imbalance/FixMatch_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:5 > /log/log-unseen-imbalance/FixMatch_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:5 > /log/log-unseen-imbalance/FixMatch_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:5 > /log/log-unseen-imbalance/FixMatch_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:5 > /log/log-unseen-imbalance/FixMatch_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo FlexMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:6 > /log/log-unseen-imbalance/FlexMatch_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FlexMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:6 > /log/log-unseen-imbalance/FlexMatch_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FlexMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:6 > /log/log-unseen-imbalance/FlexMatch_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FlexMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:6 > /log/log-unseen-imbalance/FlexMatch_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FlexMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:6 > /log/log-unseen-imbalance/FlexMatch_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo MTCF --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:6 > /log/log-unseen-imbalance/MTCF_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MTCF --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:6 > /log/log-unseen-imbalance/MTCF_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MTCF --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:6 > /log/log-unseen-imbalance/MTCF_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MTCF --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:6 > /log/log-unseen-imbalance/MTCF_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MTCF --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:6 > /log/log-unseen-imbalance/MTCF_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo Fix_A_Step --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:0 > /log/log-unseen-imbalance/Fix_A_Step_if001a.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo Fix_A_Step --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:0 > /log/log-unseen-imbalance/Fix_A_Step_if002a.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo Fix_A_Step --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:0 > /log/log-unseen-imbalance/Fix_A_Step_if005a.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo Fix_A_Step --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:0 > /log/log-unseen-imbalance/Fix_A_Step_if010a.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo Fix_A_Step --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:0 > /log/log-unseen-imbalance/Fix_A_Step_if020a.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo CAFA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:7 > /log/log-unseen-imbalance/CAFA_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo CAFA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:7 > /log/log-unseen-imbalance/CAFA_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo CAFA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:7 > /log/log-unseen-imbalance/CAFA_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo CAFA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:7 > /log/log-unseen-imbalance/CAFA_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo CAFA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:7 > /log/log-unseen-imbalance/CAFA_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo OpenMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:0 > /log/log-unseen-imbalance/OpenMatch_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo OpenMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:0 > /log/log-unseen-imbalance/OpenMatch_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo OpenMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:0 > /log/log-unseen-imbalance/OpenMatch_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo OpenMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:0 > /log/log-unseen-imbalance/OpenMatch_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo OpenMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:0 > /log/log-unseen-imbalance/OpenMatch_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo MixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:1 > /log/log-unseen-imbalance/MixMatch_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:1 > /log/log-unseen-imbalance/MixMatch_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:1 > /log/log-unseen-imbalance/MixMatch_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:1 > /log/log-unseen-imbalance/MixMatch_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo MixMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:1 > /log/log-unseen-imbalance/MixMatch_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo UDA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:2 > /log/log-unseen-imbalance/UDA_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UDA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:2 > /log/log-unseen-imbalance/UDA_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UDA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:2 > /log/log-unseen-imbalance/UDA_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UDA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:2 > /log/log-unseen-imbalance/UDA_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo UDA --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:2 > /log/log-unseen-imbalance/UDA_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo SoftMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:3 > /log/log-unseen-imbalance/SoftMatch_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo SoftMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:3 > /log/log-unseen-imbalance/SoftMatch_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo SoftMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 -device cuda:3 > /log/log-unseen-imbalance/SoftMatch_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo SoftMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:3 > /log/log-unseen-imbalance/SoftMatch_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo SoftMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:3 > /log/log-unseen-imbalance/SoftMatch_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo VAT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:4 > /log/log-unseen-imbalance/VAT_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo VAT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:4 > /log/log-unseen-imbalance/VAT_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo VAT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:4 > /log/log-unseen-imbalance/VAT_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo VAT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:4 > /log/log-unseen-imbalance/VAT_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo VAT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:4 > /log/log-unseen-imbalance/VAT_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo FreeMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:5 > /log/log-unseen-imbalance/FreeMatch_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FreeMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:5 > /log/log-unseen-imbalance/FreeMatch_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FreeMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:5 > /log/log-unseen-imbalance/FreeMatch_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FreeMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:5 > /log/log-unseen-imbalance/FreeMatch_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo FreeMatch --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:5 > /log/log-unseen-imbalance/FreeMatch_if020.log 2>&1 &

nohup python -u /evaluation_unseen_imbalance.py --algo ICT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.01 --device cuda:6 > /log/log-unseen-imbalance/ICT_if001.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo ICT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.02 --device cuda:6 > /log/log-unseen-imbalance/ICT_if002.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo ICT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.05 --device cuda:6 > /log/log-unseen-imbalance/ICT_if005.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo ICT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.10 --device cuda:6 > /log/log-unseen-imbalance/ICT_if010.log 2>&1 &
nohup python -u /evaluation_unseen_imbalance.py --algo ICT --seen_ratio 0.2 --unseen_ratio 0.4 --unseen_class_num 5 --imb_factor 0.20 --device cuda:6 > /log/log-unseen-imbalance/ICT_if020.log 2>&1 &
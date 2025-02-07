nohup python -u /evaluation_unseenchange.py --algo PseudoLabel --unseen_ratio 0 --device cuda:7 > /log/log-unseenc/PseudoLabel0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PseudoLabel --unseen_ratio 0.5 --device cuda:7 > /log/log-unseenc/PseudoLabel05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PseudoLabel --unseen_ratio 1 --device cuda:7 > /log/log-unseenc/PseudoLabel1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PseudoLabel --unseen_ratio 0.2 --device cuda:7 > /log/log-unseenc/PseudoLabel02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PseudoLabel --unseen_ratio 0.4 --device cuda:7 > /log/log-unseenc/PseudoLabel04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PseudoLabel --unseen_ratio 0.6 --device cuda:7 > /log/log-unseenc/PseudoLabel06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PseudoLabel --unseen_ratio 0.8 --device cuda:7 > /log/log-unseenc/PseudoLabel08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo PiModel --unseen_ratio 0 --device cuda:5 > /log/log-unseenc/PiModel0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PiModel --unseen_ratio 0.5 --device cuda:5 > /log/log-unseenc/PiModel05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PiModel --unseen_ratio 1 --device cuda:5 > /log/log-unseenc/PiModel1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PiModel --unseen_ratio 0.2 --device cuda:5 > /log/log-unseenc/PiModel02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PiModel --unseen_ratio 0.4 --device cuda:5 > /log/log-unseenc/PiModel04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PiModel --unseen_ratio 0.6 --device cuda:5 > /log/log-unseenc/PiModel06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo PiModel --unseen_ratio 0.8 --device cuda:5 > /log/log-unseenc/PiModel08.log 2>&1 &


nohup python -u /evaluation_unseenchange.py --algo UASD --unseen_ratio 0 --device cuda:6 > /log/log-unseenc/UASD0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UASD --unseen_ratio 0.5 --device cuda:6 > /log/log-unseenc/UASD05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UASD --unseen_ratio 1 --device cuda:6 > /log/log-unseenc/UASD1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UASD --unseen_ratio 0.2 --device cuda:6 > /log/log-unseenc/UASD02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UASD --unseen_ratio 0.4 --device cuda:6 > /log/log-unseenc/UASD04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UASD --unseen_ratio 0.6 --device cuda:6 > /log/log-unseenc/UASD06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UASD --unseen_ratio 0.8 --device cuda:6 > /log/log-unseenc/UASD08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo FixMatch --unseen_ratio 0 --device cuda:7 > /log/log-unseenc/FixMatch0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FixMatch --unseen_ratio 0.5 --device cuda:7 > /log/log-unseenc/FixMatch05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FixMatch --unseen_ratio 1 --device cuda:7 > /log/log-unseenc/FixMatch1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FixMatch --unseen_ratio 0.2 --device cuda:7 > /log/log-unseenc/FixMatch02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FixMatch --unseen_ratio 0.4 --device cuda:7 > /log/log-unseenc/FixMatch04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FixMatch --unseen_ratio 0.6 --device cuda:7 > /log/log-unseenc/FixMatch06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FixMatch --unseen_ratio 0.8 --device cuda:7 > /log/log-unseenc/FixMatch08.log 2>&1 &


nohup python -u /evaluation_unseenchange.py --algo FlexMatch --unseen_ratio 0 --device cuda:5 > /log/log-unseenc/FlexMatch0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FlexMatch --unseen_ratio 0.5 --device cuda:5 > /log/log-unseenc/FlexMatch05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FlexMatch --unseen_ratio 1 --device cuda:5 > /log/log-unseenc/FlexMatch1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FlexMatch --unseen_ratio 0.2 --device cuda:5 > /log/log-unseenc/FlexMatch02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FlexMatch --unseen_ratio 0.4 --device cuda:5 > /log/log-unseenc/FlexMatch04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FlexMatch --unseen_ratio 0.6 --device cuda:5 > /log/log-unseenc/FlexMatch06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FlexMatch --unseen_ratio 0.8 --device cuda:5 > /log/log-unseenc/FlexMatch08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo MTCF --unseen_ratio 0 --device cuda:5 > /log/log-unseenc/MTCF0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MTCF --unseen_ratio 0.5 --device cuda:5 > /log/log-unseenc/MTCF05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MTCF --unseen_ratio 1 --device cuda:5 > /log/log-unseenc/MTCF1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MTCF --unseen_ratio 0.2 --device cuda:5 > /log/log-unseenc/MTCF02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MTCF --unseen_ratio 0.4 --device cuda:5 > /log/log-unseenc/MTCF04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MTCF --unseen_ratio 0.6 --device cuda:5 > /log/log-unseenc/MTCF06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MTCF --unseen_ratio 0.8 --device cuda:5 > /log/log-unseenc/MTCF08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo CAFA --unseen_ratio 0 --device cuda:4 > /log/log-unseenc/CAFA0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo CAFA --unseen_ratio 0.5 --device cuda:4 > /log/log-unseenc/CAFA05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo CAFA --unseen_ratio 1 --device cuda:4 > /log/log-unseenc/CAFA1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo CAFA --unseen_ratio 0.2 --device cuda:4 > /log/log-unseenc/CAFA02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo CAFA --unseen_ratio 0.4 --device cuda:4 > /log/log-unseenc/CAFA04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo CAFA --unseen_ratio 0.6 --device cuda:4 > /log/log-unseenc/CAFA06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo CAFA --unseen_ratio 0.8 --device cuda:4 > /log/log-unseenc/CAFA08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo Fix_A_Step --unseen_ratio 0 --device cuda:4 > /log/log-unseenc/Fix_A_Step0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo Fix_A_Step --unseen_ratio 0.5 --device cuda:4 > /log/log-unseenc/Fix_A_Step05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo Fix_A_Step --unseen_ratio 1 --device cuda:4 > /log/log-unseenc/Fix_A_Step1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo Fix_A_Step --unseen_ratio 0.2 --device cuda:4 > /log/log-unseenc/Fix_A_Step02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo Fix_A_Step --unseen_ratio 0.4 --device cuda:4 > /log/log-unseenc/Fix_A_Step04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo Fix_A_Step --unseen_ratio 0.6 --device cuda:4 > /log/log-unseenc/Fix_A_Step06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo Fix_A_Step --unseen_ratio 0.8 --device cuda:4 > /log/log-unseenc/Fix_A_Step08.log 2>&1 &



nohup python -u /evaluation_unseenchange.py --algo OpenMatch --unseen_ratio 0 --device cuda:0 > /log/log-unseenc/OpenMatch0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo OpenMatch --unseen_ratio 0.5 --device cuda:0 > /log/log-unseenc/OpenMatch05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo OpenMatch --unseen_ratio 1 --device cuda:0 > /log/log-unseenc/OpenMatch1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo OpenMatch --unseen_ratio 0.2 --device cuda:0 > /log/log-unseenc/OpenMatch02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo OpenMatch --unseen_ratio 0.4 --device cuda:0 > /log/log-unseenc/OpenMatch04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo OpenMatch --unseen_ratio 0.6 --device cuda:0 > /log/log-unseenc/OpenMatch06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo OpenMatch --unseen_ratio 0.8 --device cuda:0 > /log/log-unseenc/OpenMatch08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo MixMatch --unseen_ratio 0 --device cuda:2 > /log/log-unseenc/MixMatch0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MixMatch --unseen_ratio 0.5 --device cuda:2 > /log/log-unseenc/MixMatch05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MixMatch --unseen_ratio 1 --device cuda:2 > /log/log-unseenc/MixMatch1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MixMatch --unseen_ratio 0.2 --device cuda:2 > /log/log-unseenc/MixMatch02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MixMatch --unseen_ratio 0.4 --device cuda:2 > /log/log-unseenc/MixMatch04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MixMatch --unseen_ratio 0.6 --device cuda:2 > /log/log-unseenc/MixMatch06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo MixMatch --unseen_ratio 0.8 --device cuda:2 > /log/log-unseenc/MixMatch08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo UDA --unseen_ratio 0 --device cuda:3 > /log/log-unseenc/UDA0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UDA --unseen_ratio 0.5 --device cuda:3 > /log/log-unseenc/UDA05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UDA --unseen_ratio 1 --device cuda:3 > /log/log-unseenc/UDA1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UDA --unseen_ratio 0.2 --device cuda:3 > /log/log-unseenc/UDA02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UDA --unseen_ratio 0.4 --device cuda:3 > /log/log-unseenc/UDA04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UDA --unseen_ratio 0.6 --device cuda:3 > /log/log-unseenc/UDA06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo UDA --unseen_ratio 0.8 --device cuda:3 > /log/log-unseenc/UDA08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo SoftMatch --unseen_ratio 0 --device cuda:4 > /log/log-unseenc/SoftMatch0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo SoftMatch --unseen_ratio 0.5 --device cuda:4 > /log/log-unseenc/SoftMatch05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo SoftMatch --unseen_ratio 1 --device cuda:4 > /log/log-unseenc/SoftMatch1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo SoftMatch --unseen_ratio 0.2 --device cuda:4 > /log/log-unseenc/SoftMatch02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo SoftMatch --unseen_ratio 0.4 --device cuda:4 > /log/log-unseenc/SoftMatch04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo SoftMatch --unseen_ratio 0.6 --device cuda:4 > /log/log-unseenc/SoftMatch06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo SoftMatch --unseen_ratio 0.8 --device cuda:4 > /log/log-unseenc/SoftMatch08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo VAT --unseen_ratio 0 --device cuda:6 > /log/log-unseenc/VAT0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo VAT --unseen_ratio 0.5 --device cuda:6 > /log/log-unseenc/VAT05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo VAT --unseen_ratio 1 --device cuda:6 > /log/log-unseenc/VAT1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo VAT --unseen_ratio 0.2 --device cuda:6 > /log/log-unseenc/VAT02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo VAT --unseen_ratio 0.4 --device cuda:6 > /log/log-unseenc/VAT04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo VAT --unseen_ratio 0.6 --device cuda:6 > /log/log-unseenc/VAT06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo VAT --unseen_ratio 0.8 --device cuda:6 > /log/log-unseenc/VAT08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo FreeMatch --unseen_ratio 0 --device cuda:5 > /log/log-unseenc/FreeMatch0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FreeMatch --unseen_ratio 0.5 --device cuda:5 > /log/log-unseenc/FreeMatch05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FreeMatch --unseen_ratio 1 --device cuda:5 > /log/log-unseenc/FreeMatch1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FreeMatch --unseen_ratio 0.2 --device cuda:5 > /log/log-unseenc/FreeMatch02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FreeMatch --unseen_ratio 0.4 --device cuda:5 > /log/log-unseenc/FreeMatch04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FreeMatch --unseen_ratio 0.6 --device cuda:5 > /log/log-unseenc/FreeMatch06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo FreeMatch --unseen_ratio 0.8 --device cuda:5 > /log/log-unseenc/FreeMatch08.log 2>&1 &

nohup python -u /evaluation_unseenchange.py --algo ICT --unseen_ratio 0 --device cuda:6 > /log/log-unseenc/ICT0.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo ICT --unseen_ratio 0.5 --device cuda:6 > /log/log-unseenc/ICT05.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo ICT --unseen_ratio 1 --device cuda:6 > /log/log-unseenc/ICT1.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo ICT --unseen_ratio 0.2 --device cuda:6 > /log/log-unseenc/ICT02.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo ICT --unseen_ratio 0.4 --device cuda:6 > /log/log-unseenc/ICT04.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo ICT --unseen_ratio 0.6 --device cuda:6 > /log/log-unseenc/ICT06.log 2>&1 &
nohup python -u /evaluation_unseenchange.py --algo ICT --unseen_ratio 0.8 --device cuda:6 > /log/log-unseenc/ICT08.log 2>&1 &
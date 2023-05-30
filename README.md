# RMTL
Code for RetailRocket Dataset. Please convert data into MDP format.

## Model Code
+ layers: stores common network structures
  + critic: critic network
  + esmm: esmm(actor) network, can introduce other MTL models as actor inside slmodels
  + layers: classical Embedding layers and MLP layers
+ slmodels: SL baseline models
+ agents: RL models
+ train: training-related configuration
+ env.py: offline sampling simulation environment
+ RLmain.py: main RL training program
+ SLmain.py: SL training main program


+ dataset
  + rtrl：retrailrocket dataset（Convert to MDP format：）[timestamp,sessionid,itemid,pay,click], [itemid,feature1,feature2,..],6:2:2

## How to run it
### MTL baselines
python3 SLmain.py --model_name=esmm

### RMTL
python3 RLmain.py
python3 SLmain.py --model_name=esmm --polish=1

Result：

test: best auc: 0.732444172986328
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:07<00:00, 19.14it/s]
task 0, AUC 0.7273702846096346, Log-loss 0.20675417715656488
task 1, AUC 0.7247954179346048, Log-loss 0.048957254763240504
   
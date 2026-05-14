HF_TOKEN='xx'

#delete repo
hf repos delete robot-learning-team43/smoll_vla_b32_60000_reward --repo-type model -y


#uploade model
hf upload robot-learning-team43/smoll_vla_b32_60000_reward  .../policy_rabc/checkpoints/060000/pretrained_model/ . --repo-type model --private


import full_process as fp

# fp.train_and_infer(mode='train1', beam_param=3, lmbda=0.5)
fp.evaluation(learned_on='train1', test_on='test1', beam_param=3)
fp.evaluation(learned_on='train1', test_on='comp1', beam_param=5)

# fp.train_and_infer(mode='train2', beam_param=3, lmbda=0.75)
fp.evaluation(learned_on='train2', test_on='comp2', beam_param=5)
# print('Process finished')

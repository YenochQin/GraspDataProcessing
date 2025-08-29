indexs -> csfs_pool_name的索引
index -> 计算时的组态索引
indexs_temp -> 所有的组态索引
stay_indexs -> 剩余的组态索引
total_data = new_desc -> 本次计算的CSFs描述符
stay_desc = desc_pool.iloc[indexs_temp] -> 所有的描述符
sum_num_min -> 计算时的组态数量
unique_indices -> 大于截断值的csfs索引
if step == 1:
    sampling_ratio = len(unique_indices)/len(index)
else:
    sampling_ratio = (len(unique_indices)-len(import_before))/len(index)
import_before = unique_indices


logger.info("             更新重要组态索引")
indexs_import_temp = index[unique_indices]
本轮计算的重要组态索引


y_stay_pred -> 推理出的组态
indexs_import_stay = np.where(y_stay_pred == 1)[0]
indexs_import_stay_temp = indexs_temp[indexs_import_stay]

np.save("results/indexs_import_ab{}_{}.npy".format(block,step-1),indexs_import_temp)
np.save("results/indexs_import_ml{}_{}.npy".format(block,step-1),indexs_import_stay_temp)
import os
import csv
import sys

def calc_avg(vals, count):
    sum = 0
    for val in vals:
        sum += float(val)
    avg = sum / count
    return str(avg)

def get_best_score(avgs, keys, is_max):
    best_key = '0'
    if is_max:
        max = 0
        for key in keys:
            if key == "評価指標" or key == 'best_key' or key == 'best_score':
                continue

            score = float(avgs[key])
            if score > max:
                max = score
                best_key = key
        return best_key
    else:
        min = 100
        for key in keys:
            if key == "評価指標" or key == 'best_key' or key == 'best_score':

                continue

            score = float(avgs[key])
            if score < min:
                min = score
                best_key = key
        return best_key


def main():
    args = sys.argv
    csv_file_path = args[1]
    print("file name is {}".format(csv_file_path))

    csv_file = open(csv_file_path, "r", encoding="ms932", errors="", newline="" )
    # 辞書形式
    f = csv.DictReader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    gaussian_key = "gaussian_filter_sd"
    nss = "NSS"
    cc = "CC"
    sim = "SIM"
    kld = "KLD"
    auc_judd = "AUC_Judd"

    nss_vals = {"0": [], "2": [], "4": [], "6": [], "8": [], "10": []}
    cc_vals = {"0": [], "2": [], "4": [], "6": [], "8": [], "10": []}
    sim_vals = {"0": [], "2": [], "4": [], "6": [], "8": [], "10": []}
    kld_vals = {"0": [], "2": [], "4": [], "6": [], "8": [], "10": []}
    auc_judd_vals = {"0": [], "2": [], "4": [], "6": [], "8": [], "10": []}

    data_count = 0

    for row in f:
        gaussian_vals = row[gaussian_key]
        if gaussian_vals == '0':
            key = '0'
            nss_vals[key].append(row[nss])
            cc_vals[key].append(row[cc])
            sim_vals[key].append(row[sim])
            kld_vals[key].append(row[kld])
            auc_judd_vals[key].append(row[auc_judd])
        elif gaussian_vals == '2':
            key = '2'
            nss_vals[key].append(row[nss])
            cc_vals[key].append(row[cc])
            sim_vals[key].append(row[sim])
            kld_vals[key].append(row[kld])
            auc_judd_vals[key].append(row[auc_judd])
        elif gaussian_vals == '4':
            key = '4'
            nss_vals[key].append(row[nss])
            cc_vals[key].append(row[cc])
            sim_vals[key].append(row[sim])
            kld_vals[key].append(row[kld])
            auc_judd_vals[key].append(row[auc_judd])
        elif gaussian_vals == '6':
            key = '6'
            nss_vals[key].append(row[nss])
            cc_vals[key].append(row[cc])
            sim_vals[key].append(row[sim])
            kld_vals[key].append(row[kld])
            auc_judd_vals[key].append(row[auc_judd])
        elif gaussian_vals == '8':
            key = '8'
            nss_vals[key].append(row[nss])
            cc_vals[key].append(row[cc])
            sim_vals[key].append(row[sim])
            kld_vals[key].append(row[kld])
            auc_judd_vals[key].append(row[auc_judd])
        elif gaussian_vals == '10':
            key = '10'
            nss_vals[key].append(row[nss])
            cc_vals[key].append(row[cc])
            sim_vals[key].append(row[sim])
            kld_vals[key].append(row[kld])
            auc_judd_vals[key].append(row[auc_judd])
            data_count += 1

    avg_nss = {'評価指標': 'NSS', 'best_key': '0', 'best_score': '', '0': '', '2': '', '4': '', '6': '', '8': '', '10': ''}
    avg_cc = {'評価指標': 'CC', 'best_key': '0', 'best_score': '', '0': '', '2': '', '4': '', '6': '', '8': '', '10': ''}
    avg_sim = {'評価指標': 'SIM', 'best_key': '0', 'best_score': '', '0': '', '2': '', '4': '', '6': '', '8': '', '10': ''}
    avg_kld = {'評価指標': 'KLD', 'best_key': '0', 'best_score': '', '0': '', '2': '', '4': '', '6': '', '8': '', '10': ''}
    avg_auc_judd = {'評価指標': 'AUC_JUDD', 'best_key': '0', 'best_score': '', '0': '', '2': '', '4': '', '6': '', '8': '', '10': ''}
    gaussian_keys = ['評価指標', 'best_key', 'best_score', '0', '2', '4', '6', '8', '10']

    # 平均の計算
    for key in gaussian_keys:
        if key == "評価指標" or key == 'best_key' or key == 'best_score':
            continue

        avg_nss[key] = calc_avg(nss_vals[key], data_count)
        avg_cc[key] = calc_avg(cc_vals[key], data_count)
        avg_sim[key] = calc_avg(sim_vals[key], data_count)
        avg_kld[key] = calc_avg(kld_vals[key], data_count)
        avg_auc_judd[key] = calc_avg(auc_judd_vals[key], data_count)

    # bestValueを算出
    nss_best_key = get_best_score(avg_nss, gaussian_keys, True)
    avg_nss['best_key'] = nss_best_key
    avg_nss['best_score'] = avg_nss[nss_best_key]

    cc_best_key = get_best_score(avg_cc, gaussian_keys, True)
    avg_cc['best_key'] = cc_best_key
    avg_cc['best_score'] = avg_cc[cc_best_key]

    sim_best_key = get_best_score(avg_sim, gaussian_keys, True)
    avg_sim['best_key'] = sim_best_key
    avg_sim['best_score'] = avg_sim[sim_best_key]

    kld_best_key = get_best_score(avg_kld, gaussian_keys, False)
    avg_kld['best_key'] = kld_best_key
    avg_kld['best_score'] = avg_kld[kld_best_key]

    auc_judd_best_key = get_best_score(avg_auc_judd, gaussian_keys, True)
    avg_auc_judd['best_key'] = auc_judd_best_key
    avg_auc_judd['best_score'] = avg_auc_judd[auc_judd_best_key]

    file_name_without_ex = os.path.splitext(os.path.basename(csv_file_path))[0]
    summarized_file_name = file_name_without_ex + "_summarized.csv"
    summarized_file_path = os.path.dirname(csv_file_path) + "/" + summarized_file_name

    with open(summarized_file_path, 'w') as f:
        writer = csv.DictWriter(f, gaussian_keys)
        writer.writeheader()
        writer.writerow(avg_nss)
        writer.writerow(avg_cc)
        writer.writerow(avg_sim)
        writer.writerow(avg_kld)
        writer.writerow(avg_auc_judd)

    print("summarized data path is {}".format(summarized_file_path))


if __name__ == "__main__":
    main()

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/transform"
)

// // 官方txt抽取出的报表位置
// var dir1 string = "/root/autodl-tmp/processed_middle_debug/txt抽取三表19到21/"

// // 我们抽取出的html的位置
// var dir2 string = "/root/autodl-tmp/processed_middle_debug/html抽取三表19到21/"

// 官方txt抽取出的报表位置
var dir1 string = "/root/autodl-tmp/processed_middle/txt抽取三表19到21/"

// 我们抽取出的html的位置
var dir2 string = "/root/autodl-tmp/processed_middle/html抽取三表19到21/"

// 输出的csv位置
var dir3 string = "/root/autodl-tmp/final_data/"

type Tuple struct {
	key   string
	value string
}

type Report struct {
	balance  []Tuple
	profit   []Tuple
	cashflow []Tuple
}

type RawRecord struct {
	TType  string          `json:"type"`
	Inside json.RawMessage `json:"inside"`
}

type Record struct {
	TType      string
	Inside     string
	InsideList []string
}

func readLine(filePath string, fileName string, map_list []string) []string {
	file, err := os.Open(filePath) // 替换为你的文件路径
	result := []string{}
	for i := 0; i < len(map_list); i++ {
		result = append(result, "0")
	}
	if err != nil {
		return result
	}

	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)

	for scanner.Scan() {
		line := scanner.Text()
		// println(line)
		key := strings.Split(line, "\001")[0]
		key = strings.Trim(key, " ")
		value := strings.Split(line, "\001")[1]
		
		if strings.Trim(value, " ") == "" {
			continue
		}
		value = strings.ReplaceAll(value, ",", "")
		value = strings.Trim(value, " ")
		f, err2 := strconv.ParseFloat(value, 64)
		if err2 != nil {
			continue
		}
		// println(result)
		for i := 0; i < len(map_list); i++ {
			s := map_list[i]
			// println(s,result[i])
			// if result[i] != "-1" {
			// 	continue
			// }
			if result[i] != "0" {
				continue
			}
			flag := true
			if !strings.Contains(s, "__") {
				match_list := strings.Split(s, "_")
				for _, v := range match_list {
					// println(key, "s1",v)
					if strings.HasPrefix(v, "^") {
						v = strings.ReplaceAll(v, "^", "")
						if v != "所有者权益" {
							if !strings.HasPrefix(key, v) {
								flag = false
								break
							}
						} else {
							if !strings.HasPrefix(key, "所有者权益") && !strings.HasPrefix(key, "股东权益") {
								flag = false
								break
							}
						}

					} else if strings.HasPrefix(v, "<") {
						v = strings.ReplaceAll(v, "<", "")
						v = strings.ReplaceAll(v, ">", "")
						if strings.Contains(key, v) {
							flag = false
							break
						}
					} else {
						if !strings.Contains(key, v) {
							flag = false
							break
						}
					}
				}
			} else {
				flag = false
				match_list := strings.Split(s, "__")
				for _, v := range match_list {
					// println(key, "s2",v)
					if strings.Contains(key, v) {
						flag = true
						break
					}
				}
			}
			// println(flag)

			if flag {
				result[i] = fmt.Sprintf("%.2f", f)
			}
		}
	}
	// println(result)
	newresult := []string{}
	for _, v := range result {
		if v != "0" {
			newresult = append(newresult, v)
		} else {
			newresult = append(newresult, "0")
		}
	}

	return newresult
}

func readLine1(filePath string, fileName string, map_list []string) []string {
	file, err := os.Open(filePath) // 替换为你的文件路径
	result := []string{}
	// for i := 0; i < len(map_list); i++ {
	// 	result = append(result, "-1")
	// }
	for i := 0; i < len(map_list); i++ {
		result = append(result, "0")
	}
	if err != nil {
		return result
	}

	defer file.Close()

	reader := transform.NewReader(file, simplifiedchinese.GBK.NewDecoder())
	scanner := bufio.NewScanner(reader)

	for scanner.Scan() {
		line := scanner.Text()
		key := strings.Split(line, "\001")[0]
		key = strings.Trim(key, " ")
		value := strings.Split(line, "\001")[1]
		if strings.Trim(value, " ") == "" {
			continue
		}
		value = strings.ReplaceAll(value, ",", "")
		value = strings.Trim(value, " ")
		f, err2 := strconv.ParseFloat(value, 64)
		if err2 != nil {
			continue
		}
		for i := 0; i < len(map_list); i++ {
			s := map_list[i]
			// if result[i] != "-1" {
			// 	continue
			// }
			if result[i] != "0" {
				continue
			}
			flag := true
			if !strings.Contains(s, "__") {
				match_list := strings.Split(s, "_")
				for _, v := range match_list {
					if strings.HasPrefix(v, "^") {
						v = strings.ReplaceAll(v, "^", "")
						if v != "所有者权益" {
							if !strings.HasPrefix(key, v) {
								flag = false
								break
							}
						} else {
							if !strings.HasPrefix(key, "所有者权益") && !strings.HasPrefix(key, "股东权益") {
								flag = false
								break
							}
						}

					} else if strings.HasPrefix(v, "<") {
						v = strings.ReplaceAll(v, "<", "")
						v = strings.ReplaceAll(v, ">", "")
						if strings.Contains(key, v) {
							flag = false
							break
						}
					} else {
						if !strings.Contains(key, v) {
							flag = false
							break
						}
					}
				}
			} else {
				flag = false
				match_list := strings.Split(s, "__")
				for _, v := range match_list {
					if strings.Contains(key, v) {
						flag = true
						break
					}
				}
			}

			if flag {
				result[i] = fmt.Sprintf("%.2f", f)
			}
		}
	}
	newresult := []string{}
	for _, v := range result {
		if v != "0" {
		// if v != "-1" {
			newresult = append(newresult, v)
		} else {
			newresult = append(newresult, "0")
		}
	}

	return newresult
}

func balance(listfile string) {
	balance_list := []string{}
	balance_list = append(balance_list, "货币资金")
	balance_list = append(balance_list, "应收票据")
	balance_list = append(balance_list, "应收利息")
	balance_list = append(balance_list, "应收账款__应收款项")
	balance_list = append(balance_list, "其他应收款")
	balance_list = append(balance_list, "预付款项")
	balance_list = append(balance_list, "存货")
	balance_list = append(balance_list, "一年内到期的非流动资产")
	balance_list = append(balance_list, "其他流动资产")
	balance_list = append(balance_list, "投资性房地产")
	balance_list = append(balance_list, "长期股权投资")
	balance_list = append(balance_list, "长期应收款")
	balance_list = append(balance_list, "固定资产")
	balance_list = append(balance_list, "工程物资")
	balance_list = append(balance_list, "在建工程")
	balance_list = append(balance_list, "无形资产")
	balance_list = append(balance_list, "商誉")
	balance_list = append(balance_list, "长期待摊费用")
	balance_list = append(balance_list, "递延所得税资产")
	balance_list = append(balance_list, "其他非流动资产")
	balance_list = append(balance_list, "短期借款")
	balance_list = append(balance_list, "应付票据")
	balance_list = append(balance_list, "应付账款")
	balance_list = append(balance_list, "预收款项")
	balance_list = append(balance_list, "应付职工薪酬_<长期>")
	balance_list = append(balance_list, "应付股利")
	balance_list = append(balance_list, "应交税费")
	balance_list = append(balance_list, "应付利息")
	balance_list = append(balance_list, "其他应付款")
	balance_list = append(balance_list, "一年内到期的非流动负债")
	balance_list = append(balance_list, "其他流动负债")
	balance_list = append(balance_list, "长期借款")
	balance_list = append(balance_list, "应付债券")
	balance_list = append(balance_list, "长期应付款")
	balance_list = append(balance_list, "预计负债")
	balance_list = append(balance_list, "递延所得税负债")
	balance_list = append(balance_list, "其他非流动负债")
	balance_list = append(balance_list, "实收资本")
	balance_list = append(balance_list, "资本公积")
	balance_list = append(balance_list, "盈余公积")
	balance_list = append(balance_list, "未分配利润")
	balance_list = append(balance_list, "其他综合收益")
	balance_list = append(balance_list, "长期应付职工薪酬")
	balance_list = append(balance_list, "长期递延收益")
	balance_list = append(balance_list, "合同资产")
	balance_list = append(balance_list, "其他非流动金融资产")
	balance_list = append(balance_list, "应付票据及应付账款")
	balance_list = append(balance_list, "合同负债")
	balance_list = append(balance_list, "其他权益工具投资")
	balance_list = append(balance_list, "负债_合计_<流动>")
	balance_list = append(balance_list, "资产_总计")
	balance_list = append(balance_list, "^所有者权益_合计")

	// balance_list = append(balance_list, "流动资产_合计")
	// balance_list = append(balance_list, "非流动资产_合计")
	// balance_list = append(balance_list, "流动负债_合计")
	// balance_list = append(balance_list, "非流动负债_合计")

	file, err := os.Open(listfile)
	if err != nil {
		log.Fatalf("failed to open file: %s", err)
	}

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	var txtlines []string

	for scanner.Scan() {
		txtlines = append(txtlines, scanner.Text())
	}

	file.Close()

	resultFileName := dir3 + "balance.csv"
	resultFile, _ := os.Create(resultFileName)
	defer resultFile.Close()
	writer := bufio.NewWriter(resultFile)
	for _, eachline := range txtlines {
		println(eachline)
		result := []string{}
		for i := 0; i < len(balance_list); i++ {
			result = append(result, "0")
		}
		// println("hello go!!!")
		// println(result["资产_总计"])
		path1 := dir1 + eachline + ".txt_balance.txt"
		s1 := readLine(path1, eachline, balance_list)
		// println(path1)

		path2 := dir2 + eachline + ".txt_balance.txt"
		s2 := readLine1(path2, eachline, balance_list)
		// println(path2)

		for i, v := range s1 {
			v1 := v
			v2 := s2[i]
			if v1 != "0" {
				result[i] = v1
			} else if v2 != "0" {
				result[i] = v2
			}
		}

		year := strings.Split(eachline, "__")[4]
		src := strings.Split(eachline, "__")[5]
		code := strings.Split(eachline, "__")[2]
		bondname := strings.Split(eachline, "__")[3]
		resultStr := strings.Join(result, "\001")
		line := fmt.Sprintf("%s\001%s\001%s\001%s\001%s", year, src, code, bondname, resultStr)
		fmt.Fprintln(writer, line)
		writer.Flush()

	}
}

func balance_static(listfile string) {
	balance_list := []string{}

	balance_list = append(balance_list, "流动资产_合计")
	balance_list = append(balance_list, "非流动资产_合计")
	balance_list = append(balance_list, "流动负债_合计")
	balance_list = append(balance_list, "非流动负债_合计")

	file, err := os.Open(listfile)
	if err != nil {
		log.Fatalf("failed to open file: %s", err)
	}

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	var txtlines []string

	for scanner.Scan() {
		txtlines = append(txtlines, scanner.Text())
	}

	file.Close()

	resultFileName := dir3 + "balance_static.csv"
	resultFile, _ := os.Create(resultFileName)
	defer resultFile.Close()
	writer := bufio.NewWriter(resultFile)
	for _, eachline := range txtlines {
		println(eachline)
		result := []string{}
		for i := 0; i < len(balance_list); i++ {
			result = append(result, "0")
		}
		path1 := dir1 + eachline + ".txt_balance.txt"
		s1 := readLine(path1, eachline, balance_list)

		path2 := dir2 + eachline + ".txt_balance.txt"
		s2 := readLine1(path2, eachline, balance_list)

		for i, v := range s1 {
			v1 := v
			v2 := s2[i]
			if v1 != "0" {
				result[i] = v1
			} else if v2 != "0" {
				result[i] = v2
			}
		}

		year := strings.Split(eachline, "__")[4]
		src := strings.Split(eachline, "__")[5]
		code := strings.Split(eachline, "__")[2]
		bondname := strings.Split(eachline, "__")[3]
		resultStr := strings.Join(result, "\001")
		line := fmt.Sprintf("%s\001%s\001%s\001%s\001%s", year, src, code, bondname, resultStr)
		fmt.Fprintln(writer, line)
		writer.Flush()

	}
}

func profit(listfile string) {
	profit_list := []string{}

	profit_list = append(profit_list, "营业总收入")
	profit_list = append(profit_list, "主营业务收入")
	profit_list = append(profit_list, "其他业务收入")
	profit_list = append(profit_list, "营业总成本")
	profit_list = append(profit_list, "主营业务成本")
	profit_list = append(profit_list, "其他业务成本")
	profit_list = append(profit_list, "税金及附加")
	profit_list = append(profit_list, "销售费用")
	profit_list = append(profit_list, "管理费用")
	profit_list = append(profit_list, "研发费用")
	profit_list = append(profit_list, "财务费用")
	profit_list = append(profit_list, "利息费用__利息支出")
	profit_list = append(profit_list, "利息收入")
	profit_list = append(profit_list, "^公允价值变动收益")
	profit_list = append(profit_list, "^投资收益")
	profit_list = append(profit_list, "营业利润")
	profit_list = append(profit_list, "营业外收入")
	profit_list = append(profit_list, "营业外支出")
	profit_list = append(profit_list, "利润总额")
	profit_list = append(profit_list, "所得税费用")
	profit_list = append(profit_list, "净利润")
	profit_list = append(profit_list, "其他收益")
	profit_list = append(profit_list, "综合收益总额")
	profit_list = append(profit_list, "基本每股收益")
	profit_list = append(profit_list, "稀释每股收益")

	file, err := os.Open(listfile)
	if err != nil {
		log.Fatalf("failed to open file: %s", err)
	}

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	var txtlines []string

	for scanner.Scan() {
		txtlines = append(txtlines, scanner.Text())
	}

	file.Close()

	resultFileName := dir3 + "profit.csv"
	resultFile, _ := os.Create(resultFileName)
	defer resultFile.Close()
	writer := bufio.NewWriter(resultFile)
	for _, eachline := range txtlines {
		println(eachline)
		result := []string{}
		for i := 0; i < len(profit_list); i++ {
			result = append(result, "0")
		}
		path1 := dir1 + eachline + ".txt_profit.txt"
		s1 := readLine(path1, eachline, profit_list)

		path2 := dir2 + eachline + ".txt_profit.txt"
		s2 := readLine1(path2, eachline, profit_list)
		if s1 == nil && s2 == nil {
			continue
		}

		for i, v := range s1 {
			v1 := v
			v2 := s2[i]
			if v1 != "0" {
				result[i] = v1
			} else if v2 != "0" {
				result[i] = v2
			}
		}

		year := strings.Split(eachline, "__")[4]
		src := strings.Split(eachline, "__")[5]
		code := strings.Split(eachline, "__")[2]
		bondname := strings.Split(eachline, "__")[3]
		resultStr := strings.Join(result, "\001")
		line := fmt.Sprintf("%s\001%s\001%s\001%s\001%s", year, src, code, bondname, resultStr)
		fmt.Fprintln(writer, line)
		writer.Flush()

	}
}

func cashFlow(listfile string) {
	cashflow_list := []string{}

	cashflow_list = append(cashflow_list, "提供劳务收到的现金")
	cashflow_list = append(cashflow_list, "收到的税费返还")
	cashflow_list = append(cashflow_list, "收到其他与经营活动有关的现金")
	cashflow_list = append(cashflow_list, "接受劳务支付的现金")
	cashflow_list = append(cashflow_list, "支付给职工以及为职工支付的现金")
	cashflow_list = append(cashflow_list, "支付的各项税费")
	cashflow_list = append(cashflow_list, "支付其他与经营活动有关的现金")
	cashflow_list = append(cashflow_list, "经营活动产生的现金流量净额")
	cashflow_list = append(cashflow_list, "收回投资收到的现金")
	cashflow_list = append(cashflow_list, "取得投资收益收到的现金")
	cashflow_list = append(cashflow_list, "无形资产和其他长期资产收回的现金净额")
	cashflow_list = append(cashflow_list, "处置子公司及其他营业单位收到的现金净额")
	cashflow_list = append(cashflow_list, "无形资产和其他长期资产支付的现金")
	cashflow_list = append(cashflow_list, "投资支付的现金")
	cashflow_list = append(cashflow_list, "支付其他与投资活动有关的现金")
	cashflow_list = append(cashflow_list, "投资活动产生的现金流量净额")
	cashflow_list = append(cashflow_list, "吸收投资收到的现金")
	cashflow_list = append(cashflow_list, "取得借款收到的现金")
	cashflow_list = append(cashflow_list, "偿还债务支付的现金")
	cashflow_list = append(cashflow_list, "利润或偿付利息支付的现金")
	cashflow_list = append(cashflow_list, "筹资活动产生的现金流量净额")
	cashflow_list = append(cashflow_list, "汇率变动对现金及现金等价物的影响")
	cashflow_list = append(cashflow_list, "期初现金及现金等价物余额")
	cashflow_list = append(cashflow_list, "期末现金及现金等价物余额")
	cashflow_list = append(cashflow_list, "现金的期末余额")

	file, err := os.Open(listfile)
	if err != nil {
		log.Fatalf("failed to open file: %s", err)
	}

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	var txtlines []string

	for scanner.Scan() {
		txtlines = append(txtlines, scanner.Text())
	}

	file.Close()

	resultFileName := dir3 + "cashflow.csv"
	resultFile, _ := os.Create(resultFileName)
	defer resultFile.Close()
	writer := bufio.NewWriter(resultFile)
	for _, eachline := range txtlines {
		println(eachline)
		result := []string{}
		for i := 0; i < len(cashflow_list); i++ {
			result = append(result, "0")
		}
		path1 := dir1 + eachline + ".txt_cashflow.txt"
		s1 := readLine(path1, eachline, cashflow_list)

		path2 := dir2 + eachline + ".txt_cashflow.txt"
		s2 := readLine1(path2, eachline, cashflow_list)
		if s1 == nil && s2 == nil {
			continue
		}

		for i, v := range s1 {
			v1 := v
			v2 := s2[i]
			if v1 != "0" {
				result[i] = v1
			} else if v2 != "0" {
				result[i] = v2
			}
		}

		year := strings.Split(eachline, "__")[4]
		src := strings.Split(eachline, "__")[5]
		code := strings.Split(eachline, "__")[2]
		bondname := strings.Split(eachline, "__")[3]
		resultStr := strings.Join(result, "\001")
		line := fmt.Sprintf("%s\001%s\001%s\001%s\001%s", year, src, code, bondname, resultStr)
		fmt.Fprintln(writer, line)
		writer.Flush()

	}

}

func main() {
	listfile := "./output.txt"
	// print(listfile)
	balance(listfile)
	balance_static(listfile)
	cashFlow(listfile)
	profit(listfile)
}

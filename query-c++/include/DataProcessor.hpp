#ifndef LEARNED_INDEX_DATAPROCESSOR_H
#define LEARNED_INDEX_DATAPROCESSOR_H


#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <ctime>
#include <chrono>
//#include "cnpy.h"


#include "Utilities.hpp"


class DataProcessor {
private:
    std::string filename;
public:
    virtual std::vector<key_type> read_data_csv() {
        return std::vector<key_type>();
    }

};


// Transform the timestamp into concatenated get_payload_given_key.
// The format is "2017-01-17 00:00:00"
static inline std::string TimestampTrim(std::string &str) {
    str.erase(std::remove(str.begin(), str.end(), '-'), str.end());
    str.erase(std::remove(str.begin(), str.end(), ':'), str.end());
    str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
    return str;
}


// The format of Iot data: timestamp, device_id, device_type, device_name, device_floor, event_type, event_value
class IotProcessor : public DataProcessor {
public:

    std::vector<key_type> read_data_binary(const std::string &filename,
                                           size_t max_lines = 1000000000) {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Begin to load data, the dataset file path:" << filename << std::endl;

        std::ifstream fin;
        fin.open(filename, std::ios::binary);
        std::vector<key_type> data;
        if (!fin) {
            std::cout << "open file failure." << std::endl;
            exit(1);
        }
        size_t data_size;
        fin.read((char *) &data_size, sizeof(int64_t));
        data.reserve(data_size);

        if (data_size > max_lines) {
            std::cout << "too large data or wrong data size: " << data_size << std::endl;
            exit(1);
        }

        int distinct_i = 1;
        key_type key_i, last_key;
        fin.read((char *) &last_key, sizeof(double));
        data.emplace_back(last_key);
        for (size_t i = 1; i < data_size; i++) {
            if (i % 500000 == 0) std::cout << "Read line: " << i << ". Disticnt keys:" << distinct_i << std::endl;
            fin.read((char *) &key_i, sizeof(int64_t));
            if (last_key != key_i) {
                data.emplace_back(key_i);
                distinct_i++;
            }
            last_key = key_i;
            //fin.read((char *) &data[i], sizeof(int64_t));
        }
        std::cout << "Finished to read the binary file." << std::endl;

        return data;
    }

    std::vector<key_type> read_data_csv(const std::string &filename,
                                        size_t max_lines = std::numeric_limits<size_t>::max()) {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Begin to load data, the dataset file path:" << filename << std::endl;
        std::ifstream fin(filename);
        std::string line;
        std::vector<key_type> data;
        size_t max_data_length = max_lines == std::numeric_limits<size_t>::max() ? 1024 : max_lines;
        if (!fin) {
            std::cout << "open file failure." << std::endl;
            exit(1);
        }
        data.reserve(max_data_length);
        //size_t last_query = 0;
        while (getline(fin, line)) {
            std::istringstream sin(line);
            //std::vector<std::string> fields;
            std::string field;
            getline(sin, field, ',');
            //while (getline(sin, field, ',')){
            //fields.push_back(field);
            //}
            //fields.push_back(field);
            //size_t get_payload_given_key = std::stoll(TimestampTrim(fields[0]));
            key_type query = std::stoll(TimestampTrim(field));
            data.push_back(query);
            //if (query != last_query) {data.push_back(get_payload_given_key);}
            //last_query = get_payload_given_key;  // distinct, follow the setting of ``the case for learned index structures, SIGMOD 18''
        }
        end = std::chrono::system_clock::now();
        end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Load data finished." << std::endl;

        return data;
    }


};


// The original format of Weblog data: "253.224.146.208,,2016-11-03 00:00:00,GET,/people/0/tagthat.jpg,HTTP/1.1,304,0"
class WebBlogsProcessor : public DataProcessor {
public:

    std::vector<key_type> read_data_binary(const std::string &filename,
                                           size_t max_lines = std::numeric_limits<size_t>::max()) {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Begin to load data, the dataset file path:" << filename << std::endl;

        std::ifstream fin;
        fin.open(filename, std::ios::binary);
        std::vector<key_type> data;
        if (!fin) {
            std::cout << "open file failure." << std::endl;
            exit(1);
        }
        size_t data_size;
        fin.read((char *) &data_size, sizeof(int64_t));

        if (data_size > max_lines) {
            std::cout << "too large data or wrong data size: " << data_size << std::endl;
            exit(1);
        }

        data.reserve(data_size);


        int distinct_i = 1;
        key_type key_i, last_key;
        fin.read((char *) &last_key, sizeof(double));
        data.emplace_back(last_key);
        for (size_t i = 1; i < data_size; i++) {
            if (i % 500000 == 0) std::cout << "Read line: " << i << ". Disticnt keys:" << distinct_i << std::endl;
            fin.read((char *) &key_i, sizeof(int64_t));
            if (last_key != key_i) {
                data.emplace_back(key_i);
                distinct_i++;
            }
            last_key = key_i;
            //fin.read((char *) &data[i], sizeof(int64_t));
        }
        std::cout << "Finished to read the binary file." << std::endl;

        return data;
    }

    std::vector<key_type> read_data_csv(const std::string &filename,
                                        size_t max_lines = std::numeric_limits<size_t>::max()) {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Begin to load data, the dataset file path:" << filename << std::endl;
        std::ifstream fin(filename);
        std::string line;
        std::vector<key_type> data;
        size_t max_data_length = max_lines == std::numeric_limits<size_t>::max() ? 1024 : max_lines;
        if (!fin) {
            std::cout << "open file failure." << std::endl;
            exit(1);
        }
        data.reserve(max_data_length);
        size_t last_query = 0;
        long count_i = 0, load_count_i = 0;
        while (getline(fin, line)) {
            if (count_i % 200000 == 0)
                std::cout << "Read line: " << count_i << ", load data count: " << load_count_i << std::endl;
            std::istringstream sin(line);
            std::vector<std::string> fields;
            std::string field;
            while (getline(sin, field, ',')) {
                fields.push_back(field);
            }
            //getline(sin, field, ','); // skip the ip
            //getline(sin, field, ','); // skip the second ','
            field = fields[2];
            size_t query = std::stoll(TimestampTrim(field));
            if (query != last_query) {
                data.push_back(query);
                load_count_i++;
            }
            last_query = query;// distinct, follow the setting of ``the case for learned index structures, SIGMOD 18''
            count_i++;
            //if (count_i == 1000000) return data;
        }
        end = std::chrono::system_clock::now();
        end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Load data finished." << std::endl;
        return data;
    }

};


class MapProcessor : public DataProcessor {
public:

    void read_data_binary(std::vector<key_type> &data, const std::string &filename,
                          size_t max_lines = std::numeric_limits<size_t>::max(),
                          double data_size = -1, bool de_duplicate=true) {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Begin to load data, the dataset file path:" << filename << std::endl;

        std::ifstream fin;
        fin.open(filename, std::ios::binary);
        if (!fin) {
            std::cout << "open file failure." << std::endl;
            exit(1);
        }
        if (data_size < 0) // the first element in the file is data size, i.e., has header
        {
            fin.read((char *) &data_size, sizeof(double));
        }
        if (data_size > max_lines) {
            std::cout << "too large data or wrong data size: " << data_size << std::endl;
            exit(1);
        }


        data.reserve(int(data_size));

        int distinct_i = 1;
        double key_i, last_key;
        fin.read((char *) &last_key, sizeof(double));
        data.emplace_back(last_key);
        for (size_t i = 1; i < data_size; i++) {
            if (i % 500000 == 0) std::cout << "Read line: " << i << ". Disticnt keys:" << distinct_i << std::endl;
            fin.read((char *) &key_i, sizeof(double));
            if (not de_duplicate or (de_duplicate and last_key != key_i)) {
                data.emplace_back(key_i);
                distinct_i++;
            }
            last_key = key_i;
            //fin.read((char *) &data[i], sizeof(int64_t));
        }
        std::cout << "Finished to read the binary file." << std::endl;
    }

};


// The format of Lognormal data
class LognormalProcessor : public DataProcessor {
public:

    /*
     * generate synthetic lognormal dataset, save the binary format into file_path
     */
    void generate_data_save_binary(size_t data_size, const std::string &file_path = "./lognormal.bin",
                                   double u = 0.0, double sigma = 2.0, bool round_value = true, double timer=1e9) {
        std::vector<key_type> generated_data;
        generated_data.reserve(data_size);

        std::default_random_engine generator;
        std::lognormal_distribution<double> distribution(u, sigma);
        std::cout << "Begin to load data, the dataset file path:" << file_path << ", round_value: " << round_value
                  << std::endl;
        long count_i = 0;
        for (size_t i = 0; i < data_size + 1; i++) {
            key_type key;
            if (round_value) {
                key = round(distribution(generator) * timer);
            } else {
                key = distribution(generator);
            }
            generated_data.emplace_back(key);
            if (count_i % 1000000 == 0) std::cout << "Generation count: " << count_i << std::endl;
            count_i++;
        }

        std::cout << "Begin to sort data" << std::endl;
        std::sort(generated_data.begin(), generated_data.end());

        write_vector_to_f<key_type>(generated_data, file_path);

    }


    std::vector<key_type> read_data_binary(const std::string &filename,
                                           size_t max_lines = std::numeric_limits<size_t>::max()) {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Begin to load data, the dataset file path:" << filename << std::endl;

        std::ifstream fin;
        fin.open(filename, std::ios::binary);
        std::vector<key_type> data;
        if (!fin) {
            std::cout << "open file failure." << std::endl;
            exit(1);
        }
        key_type data_size;
        fin.read((char *) &data_size, sizeof(key_type));
        if (data_size > max_lines) {
            std::cout << "too large data or wrong data size: " << data_size << std::endl;
            exit(1);
        }

        data.reserve(data_size);

        int distinct_i = 1;
        key_type key_i, last_key;
        fin.read((char *) &last_key, sizeof(double));
        data.emplace_back(last_key);
        for (size_t i = 0; i < data_size; i++) {
            if (i % 1000000 == 0) std::cout << "Read line: " << i << ". Disticnt keys:" << distinct_i << std::endl;
            fin.read((char *) &key_i, sizeof(key_type));
            if (not isEqual(last_key, key_i)) {
                data.emplace_back(key_i);
                distinct_i++;
            }
            last_key = key_i;
        }
        std::cout << "Finished to read the binary file." << std::endl;

        return data;
    }


};


// The format of EVT (ExtremelyVariedInterval) data
class ExtremelyVariedIntervalProcessor : public DataProcessor {
public:

    /*
     * generate synthetic Extremely Varied Interval dataset, save the binary format into file_path
     */
    void generate_data_save_binary(size_t repeated_times, const std::string &file_path = "./evt.bin",
                                   long n_of_moderate_data = 3, long epsilon = 100) {
        srand(1234);
        std::vector<key_type> generated_data;
        generated_data.reserve(repeated_times);

        std::cout << "Begin to load data, the dataset file path:" << file_path << std::endl;
        key_type x_cur = 0;
        for (size_t i = 0; i < repeated_times; i++) {
            // moderate density
            for (int j = 0; j < n_of_moderate_data; j++) {
                generated_data.emplace_back(x_cur);
                x_cur += double(epsilon) / 2;
            }
            // radical density
            x_cur += 1.0 / double(epsilon);
            for (int i = 0; i < epsilon + 1; i++) {
                generated_data.emplace_back(x_cur);
                x_cur += 0.000001;
            }
            //breaking node
            x_cur += 1.0 / double(epsilon);
            generated_data.emplace_back(x_cur);
        }

        std::cout << "Begin to sort data" << file_path << std::endl;
        std::sort(generated_data.begin(), generated_data.end());

        write_vector_to_f<key_type>(generated_data, file_path);

        // cnpy::npy_save("../tmp_out/toy_data.npy", data_type_transformed);
        // std::cout<< "Save toy data to " << "../tmp_out/toy_data.npy" <<std::endl;

    }


    std::vector<key_type> read_data_binary(const std::string &filename,
                                           size_t max_lines = std::numeric_limits<size_t>::max()) {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << std::ctime(&end_time) << "Begin to load data, the dataset file path:" << filename << std::endl;

        std::ifstream fin;
        fin.open(filename, std::ios::binary);
        std::vector<key_type> data;
        if (!fin) {
            std::cout << "open file failure." << std::endl;
            exit(1);
        }
        size_t data_size;
        fin.read((char *) &data_size, sizeof(key_type));
        if (data_size > max_lines) {
            std::cout << "too large data or wrong data size: " << data_size << std::endl;
            exit(1);
        }

        data.reserve(data_size);

        int distinct_i = 1;
        key_type key_i, last_key;
        fin.read((char *) &last_key, sizeof(double));
        data.emplace_back(last_key);
        for (size_t i = 1; i < data_size; i++) {
            if (i % 1000000 == 0) std::cout << "Read line: " << i << ". Disticnt keys:" << distinct_i << std::endl;
            fin.read((char *) &key_i, sizeof(key_type));
            if (last_key != key_i) {
                data.emplace_back(key_i);
                distinct_i++;
            }
            last_key = key_i;
        }
        std::cout << "Finished to read the binary file." << std::endl;

        return data;
    }


};


#endif //LEARNED_INDEX_DATAPROCESSOR_H

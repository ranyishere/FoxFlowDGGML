#ifndef LISTENERS_HPP
#define LISTENERS_HPP

#include <iostream>
#include <string>
#include "vtu11/vtu11.hpp"
#include <vector>


namespace DGGML {

    class Listener {
    public:

        int header_length = 0;
        void setFileName(const std::string& filename) { this->file_name = filename; }

        std::string getFileName(){ return this->file_name; }

        void createCSVFile() {
            std::string file_name = this->file_name + ".csv";
            std::ofstream outfile(file_name);
        }

        void writeHeader(const std::vector<std::string>& headers) {

            // Store header length
            header_length = headers.size();

            std::string file_name = this->file_name + ".csv";
            std::ofstream file(file_name, std::ios::out | std::ios::app);

            if (file.is_open()) {
                for (size_t i = 0; i < headers.size(); ++i) {
                    file << headers[i];
                    if (i < headers.size() - 1) {
                        file << ",";
                    }
                }
                file << "\n";
                file.close();
            } else {
                std::cerr << "Error opening file: " << file_name << "\n";
            }

        }

        void writeDataRow(const std::vector<std::string>& data) {

            // Make sure length of data matches header length
            if (data.size() != header_length) {
                std::cerr << "Data length does not match header length!\n";
                exit(1);
                return;
            }

            std::string file_name = this->file_name + ".csv";
            std::ofstream file(file_name, std::ios::out | std::ios::app);

            if (file.is_open()) {
                for (size_t i = 0; i < data.size(); ++i) {
                    file << data[i];
                    if (i < data.size() - 1) {
                        file << ",";
                    }
                }
                file << "\n";
                file.close();
            } else {
                std::cerr << "Error opening file: " << file_name << "\n";
            }
        }

        virtual void onUpdate() = 0;
        virtual void getInfo() {}
        virtual ~Listener() = default;

    protected:
        std::string file_name = "default_listener";
    };

}
    
#endif

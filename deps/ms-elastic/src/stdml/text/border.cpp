#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::string readlines(std::istream &in)
{
    std::string text;
    for (std::string line; std::getline(in, line);) {
        text += line + '\n';
    }
    return text;
}

std::string readfile(const char *filename)
{
    std::ifstream in(filename);
    return readlines(in);
}

std::string rtrim(std::string s)
{
    for (; !s.empty() && std::isspace(*s.rbegin()); s.erase(s.end() - 1))
        ;
    return s;
}

std::string et(std::string s, int i = 4)
{
    std::stringstream ss;
    for (auto c : s) {
        if (c == '\t') {
            ss << std::string(i, ' ');
        } else {
            ss << c;
        }
    }
    return ss.str();
}

std::string border(std::string text, std::string title)
{
    std::vector<std::string> lines;
    std::stringstream ss(text);
    size_t w = 0;
    for (std::string line; std::getline(ss, line);) {
        line = rtrim(std::move(line));
        line = et(std::move(line));
        if (line.size() > w) {
            w = line.size();
        }
        lines.push_back(line);
    }
    std::string hr(w, '-');
    for (auto &line : lines) {
        line.resize(w, ' ');
        line = "| " + line + " |";
    }
    title.resize(hr.size(), '-');
    std::string s = "+-" + title + "-+";
    lines.push_back("+-" + hr + "-+");
    for (auto &line : lines) {
        s += "\n" + line;
    }
    return s;
}

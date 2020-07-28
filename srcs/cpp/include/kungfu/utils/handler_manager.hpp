#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <thread>

template <typename handle_t = int>
class HandleManager
{
    using state_t = std::atomic<bool>;

    std::atomic<handle_t> counter_;
    std::map<handle_t, state_t *> handles_;
    std::mutex mu_;

  public:
    handle_t create()
    {
        handle_t handle = counter_++;
        state_t *state  = new state_t(false);
        {
            std::lock_guard<std::mutex> lk(mu_);
            handles_[handle] = state;
        }
        return handle;
    }

    void done(handle_t handle)
    {
        state_t *state = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            state = handles_.at(handle);
        }
        state->store(true);
    }

    void wait(handle_t handle)
    {
        state_t *state = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            state = handles_.at(handle);
        }
        while (!state->load()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(50));
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            handles_.erase(handle);
        }
        delete state;
    }

    void wait_all(const std::vector<handle_t> &handles)
    {
        if (handles.empty()) { return; }
        const int n = handles.size();
        std::vector<state_t *> states(n);
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; ++i) { states[i] = handles_.at(handles[i]); }
        }
        std::vector<bool> finished(n);
        std::fill(finished.begin(), finished.end(), false);
        for (;;) {
            bool all_finished = true;
            for (int i = 0; i < n; ++i) {
                if (finished[i]) { continue; }
                all_finished = false;
                if (states[i]->load()) { finished[i] = true; }
            }
            if (all_finished) { break; }
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto h : handles) { handles_.erase(h); }
        }
        for (int i = 0; i < n; ++i) { delete states[i]; }
    }
};

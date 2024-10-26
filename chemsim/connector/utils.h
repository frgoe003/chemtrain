//
// Created by Paul Fuchs on 26.10.24.
//

#ifndef UTILS_H
#define UTILS_H


namespace jcn {

    class DEBUGSettings {
        public:
            DEBUGSettings() {
                char* ll = getenv("JCN_LOG_LEVEL");
                if (ll != nullptr) {
                    log_level = std::stoi(ll);
                }
            };

            int log_level = 0;
    }


} // namespace jcn



#endif //UTILS_H

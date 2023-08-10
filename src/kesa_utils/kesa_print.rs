use owo_colors::OwoColorize;

#[derive(Debug, Clone)]
pub enum KesaLogType {
    Warn,
    Info,
    Wafak,
}

#[derive(Debug, Clone)]
pub struct KesaLog {
    pub message: String,
    pub logtype: KesaLogType,
}

impl KesaLog {
    pub fn KesaLogWarn(msg: String) -> KesaLog {
        let log_type = String::from("[Warning]: ");
        println!("{} {}", log_type.white().on_yellow(), &msg);
        KesaLog {
            message: msg,
            logtype: KesaLogType::Warn,
        }
    }

    pub fn KesaLogInfo(msg: String) -> KesaLog {
        let log_type = String::from("[Info]: ");
        println!("{} {}", log_type.white().on_green(), &msg);
        KesaLog {
            message: msg,
            logtype: KesaLogType::Info,
        }
    }
}

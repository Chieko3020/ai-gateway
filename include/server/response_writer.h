// 响应写出器的前置声明：路由与连接处理器只需要引用，不需要完整定义。
// 完整定义在 server/http_server.h（ResponseWriter 与 HttpServer 同属"服务器
// 写路径"这一层，放在一起是为了让 send_all 的死线语义只有一处说明）
#pragma once

namespace ai_gateway {

class ResponseWriter;

}  // namespace ai_gateway

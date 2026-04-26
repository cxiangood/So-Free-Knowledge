# 个人助理聚合简报

生成时间：2026-04-26

## 当前状态
- 已完成在线卡片推送链路验证。
- 飞书在线建文档失败：缺少 `docx:document` / `docx:document:create` 应用权限。
- 在线拉群消息也受限：缺少 `im:message:readonly`（应用身份）。

## 建议
1. 开通 `docx:document:create` 后可自动创建飞书文档并回填文档链接到卡片。
2. 开通 `im:message:readonly` 后可在个人助理里自动聚合群聊语料并打分。

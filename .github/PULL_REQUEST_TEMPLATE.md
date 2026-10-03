## 改动说明 / What does this PR do?

<!-- 一两句话说明改了什么、为什么。 -->

## 改动范围 / Scope

- [ ] 引擎 `engine/`（若改动，需运行 `python scripts/_sync_engine.py` 同步到 `plugin/engine/`）
- [ ] 工具表 `plugin/lib/tools.js`（若改动，需更新 `scripts/_specs.json` 并跑契约校验）
- [ ] 文档 `docs/`、`README.md` / `README.en.md`
- [ ] 其它

## 自测 / How has this been tested?

- [ ] `python -X utf8 tests/_test_acceptance.py`（111 例全部通过）
- [ ] `python -X utf8 tests/_test_special.py`（8/8 通过）
- [ ] `python -X utf8 scripts/_check_contract.py`（契约校验通过，仅当改了工具表/引擎签名）
- [ ] 新增了可复现的测试用例（新增能力/修缺陷时）

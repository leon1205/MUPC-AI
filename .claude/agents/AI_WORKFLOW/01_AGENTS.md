# AI 团队组织架构与花名册

> 本文件是 AI 团队的总索引。所有角色协作必须遵循 `02_WORKFLOW.md` 定义的合同驱动流程，并使用 `05_GLOSSARY.md` 的统一术语。

## 团队角色清单


| 序号 | 角色名称                           | 核心职责                                                                                  | 对应文档                            |
| :--- | :--------------------------------- | :---------------------------------------------------------------------------------------- | :---------------------------------- |
| 1    | **项目经理 (Manager)**             | **流程路由与门禁控制者**。选择项目路径，调度专家，检查阶段交付物标记，决定流程推进。      | `.claude/agents/manager.md`         |
| 2    | **需求分析师 (PRD Writer)**        | **需求定义者**。将模糊需求转化为逻辑严密、可供评审的PRD文档 (`[REVIEWED: PASS]`)。        | `.claude/agents/prd_writer.md`      |
| 3    | **需求评审员 (PRD Reviewer)**      | **需求守门人**。评审PRD的完整性、清晰度与可行性，决定是否放行至设计阶段。                 | `.claude/agents/prd_reviewer.md`    |
| 4    | **架构师 (Architect)**             | **技术设计者**。基于PRD，进行技术选型与系统设计，产出技术设计文档 (`[DESIGN_APPROVED]`)。 | `.claude/agents/architect.md`       |
| 5    | **设计评审员 (Design Reviewer)**   | **设计质检员**。评审技术设计的安全性、合理性与完整性。                                    | `.claude/agents/design_reviewer.md` |
| 6    | **UI/UX 设计师 (UI/UX Designer)**  | **界面设计者**。基于PRD，进行视觉与交互设计，产出UI设计文档 (`[DESIGN_APPROVED]`)。       | `.claude/agents/ui_ux_designer.md`  |
| 7    | **UI 评审员 (UI Reviewer)**        | **界面艺术指导**。评审UI设计稿的还原度、一致性与用户体验。                                | `.claude/agents/ui_reviewer.md`     |
| 8    | **开发工程师 (Developer)**         | **代码实现者**。严格按照通过评审的设计文档进行编码。                                      | `.claude/agents/developer.md`       |
| 9    | **代码评审员 (Code Reviewer)**     | **代码守门员**。评审代码的安全性、规范性与质量，产出评审报告 (`[CODE_REVIEWED: PASS]`)。  | `.claude/agents/code_reviewer.md`   |
| 10   | **自动化测试工程师 (QA Engineer)** | **质量验证者**。编写并执行测试，确保功能符合PRD，产出测试报告 (`[TEST_PASSED]`)。         | `.claude/agents/qa_engineer.md`     |

## 核心协作流程 (合同驱动)

1. **需求合同**：Manager -(派发)-> PRD Writer -(产出)-> PRD -(送审)-> PRD Reviewer -(标记)-> `[REVIEWED: PASS]`
2. **设计合同**：Manager -(派发)-> Architect/UI Designer -(产出)-> 设计文档 -(送审)-> Design/UI Reviewer -(标记)-> `[DESIGN_APPROVED]`
3. **开发与门禁**：Manager -(派发)-> Developer -(编码)-> 代码 -(送审)-> Code Reviewer -(标记)-> `[CODE_REVIEWED: PASS]` -(送测)-> QA Engineer -(标记)-> `[TEST_PASSED]`
4. **交付**：Manager -(确认)-> 交付用户

**关键**：Manager 仅在收到正确标记后，才会派发下一阶段任务。

## 维护说明

* 当需要新增或修改角色时，请同步更新本表格及 `.claude/agents/` 下的对应文件。
* 任何 Agent 在启动时，应先读取本文件以了解团队全局结构。

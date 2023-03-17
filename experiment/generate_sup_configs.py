import dataclasses
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Any
from pyhocon import ConfigTree, HOCONConverter


@dataclasses.dataclass
class Variable:
    name: str
    default: Any
    override: Any

    def values(self):
        return {"+": self.default, "-": self.override}.items()


var1 = Variable("model.disable_reconstruction", False, True)
var2 = Variable("load_model", True, False)
var3 = Variable("fit_if_no_malware", True, False)
var4 = Variable(
    "client.client_malware",
    """{
          1: [M1, M1, _, M1, M1]
          2: [M2, _, M2, M2, M2]
          3: [M3, M3, M3, M3, M3]
          4: [_, M4, M4, M4, M4]
          5: [M5, M5, M5, M5, M5]
          6: [M6, M6, M6, _, M6]
        }""",
    """{
      1: [_, M1_2, ____, M1_4, M1_5]
      2: [_, ____, ____, M2_4, M2_5]
      3: [_, ____, ____, ____, M3_1]
      4: [_, ____, M4_3, M4_4, M4_5]
      5: [_, M5_1, ____, ____, ____]
      6: [_, ____, M6_3, ____, M6_5]
    }""",
)

config_prefix = "exp_sup_"


def main():
    sup_dir = Path("../supervised_detection")
    for v1, v1_value in var1.values():
        config = {}
        config[var1.name] = v1_value
        exp_id = f"{config_prefix}v1_{v1_value}"
        config["id"] = exp_id
        write_conf(config, sup_dir / f"{exp_id}.conf")

        config_ = deepcopy(config)
        exp_id_ = f"{exp_id}_v2_{var2.override}"
        config_[var2.name] = var2.override
        config_["id"] = exp_id_
        write_conf(config_, sup_dir / f"{exp_id_}.conf")

        config_ = deepcopy(config)
        exp_id_ = f"{exp_id}_v3_{var3.override}"
        config_[var3.name] = var3.override
        config_["id"] = exp_id_
        write_conf(config_, sup_dir / f"{exp_id_}.conf")

        for v2, v2_value in var2.values():
            config_ = deepcopy(config)
            exp_id_ = f"{exp_id}_v2_{v2_value}_v4_scenario"
            config_[var2.name] = v2_value
            config_[var4.name] = var4.override
            config_["id"] = exp_id_
            write_conf(config_, sup_dir / f"{exp_id_}.conf")


def write_conf(config, file):
    with file.open("w") as f:
        f.write("{\n")
        for key, value in config.items():
            f.write(f"\t{key}: {value}\n")
        f.write("}\n")


if __name__ == "__main__":
    main()

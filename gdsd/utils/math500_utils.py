import re


def boxed_in_answer(prompts, completions, answer, step=None, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        reward = 0.0
        try:
            answer_text = response.split("<answer>", 1)[1].split("</answer>", 1)[0]
            reward += 1.0
        except Exception:
            answer_text = response
        reward += 1.0 if "\\boxed" in answer_text else 0.5
        rewards.append(reward)
    return rewards


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if s is None:
        return None

    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left) :] if s.startswith(left) else s

    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(idx, len(string)):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) <= 1:
        return string

    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr:
            return string
        if substr[0] == "{":
            new_str += substr
            continue

        if len(substr) < 2:
            return string
        a = substr[0]
        b = substr[1]
        post_substr = substr[2:] if len(substr) > 2 else ""
        if b != "{":
            new_str += "{" + a + "}{" + b + "}" + post_substr
        else:
            new_str += "{" + a + "}" + b + post_substr
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
    except ValueError:
        return string
    if string != f"{a_int}/{b_int}":
        return string
    return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"


def remove_right_units(string):
    if "\\text{ " in string:
        return string.split("\\text{ ", 1)[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
        elif split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string

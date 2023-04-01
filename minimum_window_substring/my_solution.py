from collections import OrderedDict


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        letters_dict = OrderedDict()
        for l in t:
            if l not in letters_dict:
                letters_dict[l] = [1, []]
            else:
                letters_dict[l][0] += 1

        for i, l in enumerate(s):
            if l in letters_dict:
                letters_dict[l][1].append(i)

        indices = [0 for letter in letters_dict.values()]
        minimum_length = 10 ** 5 + 1
        minimum_start_window = 0
        minimum_end_window = 10 ** 5
        while True:
            try:
                current_indices = [letter_indices[1][indices[i]] for i, letter_indices in
                                   enumerate(letters_dict.values())]
                last_possible_index = [letter_indices[1][indices[i] + letter_indices[0] - 1] for i, letter_indices in
                                       enumerate(letters_dict.values())]
            except:
                break
            window_start_index = min(*current_indices) if len(current_indices) != 1 else current_indices[0]
            window_end_index = max(*last_possible_index) if len(current_indices) != 1 else last_possible_index[0]
            current_length = window_end_index - window_start_index + 1
            if current_length < minimum_length:
                minimum_length = current_length
                minimum_start_window = window_start_index
                minimum_end_window = window_end_index
            if indices[current_indices.index(window_start_index)] + 1 == \
                    len(list(letters_dict.values())[current_indices.index(window_start_index)][1]):
                break
            indices[current_indices.index(window_start_index)] += 1

        if minimum_length == 10 ** 5 + 1:
            return ""
        return s[minimum_start_window: minimum_end_window + 1]

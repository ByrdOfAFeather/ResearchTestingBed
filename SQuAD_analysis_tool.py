import json
import tkinter as tk
import tkinter.ttk as ttk


class MainGUI(tk.Frame):
	def __init__(self, master, question_answer_text):
		super(MainGUI, self).__init__()
		self.master = master
		self.frame = None
		self.data = question_answer_text
		self.selections = []
		self._org_data_length = len(self.data)
		self.selected = 0
		self.options_variable = tk.StringVar(self.master)
		self.reasons = []
		self.reasons_entry = tk.StringVar(self.master)
		master.title("A simple GUI")

		self.label = ttk.Label(master, text="Select the Most Difficult Question")
		self.label.pack()
		self.build_selection()

	def build_selection(self):
		items = self._sample()
		if self.frame is None:
			self.frame = tk.Frame(self.master)
			self.frame.pack()
		prev = None
		for i in items:
			if prev != i[0]:
				tk.Label(self.frame, text=i[0], wraplength=500).pack()
			prev = i[0]
			ttk.Button(self.frame, text=i[3], command=lambda *args: self._handle_choice(i)).pack()
			ttk.Label(self.frame, text=i[1]).pack()

	def _handle_options_choice(self, choice):
		self.reasons.append(self.options_variable.get())
		saveable = list(choice)
		saveable.append(self.options_variable.get())
		self.selections.append(saveable)
		self.option_menu.destroy()
		self.frame.destroy()
		self.frame = None
		_temp = {"selections": self.selections, "saved": self.selected}
		with open("back.json", "w") as f:
			json.dump(_temp, f)
		self.build_selection()

	def _handle_reason_button_proxy(self):
		self.options_variable.set(self.reasons_entry.get())

	def _handle_choice(self, choice):
		self.options_variable = tk.StringVar(self.master)
		self.reasons_entry = tk.StringVar(self.master)
		self.option_menu = ttk.OptionMenu(self.frame, self.options_variable, f"Reason", *self.reasons)
		self.option_menu.pack()
		self.options_variable.trace('w', lambda *args: self._handle_options_choice(choice))
		tk.Entry(self.frame, textvariable=self.reasons_entry).pack()
		ttk.Button(self.frame, text="Save", command=lambda *args: self._handle_reason_button_proxy()).pack()

	def _sample(self):
		self.selected += 5
		return [self.data.pop() for _ in range(0, 5)]


data_json = json.load(open(f'data/train-v2.0.json', 'r'))
overall_qas_idx = 0
question_context_answer = []
for overall_idx, _ in enumerate(data_json['data']):
	for paragraphs in data_json['data'][overall_idx]['paragraphs']:
		for qas_idx, question_answer in enumerate(paragraphs['qas']):
			if question_answer["is_impossible"]:
				continue
			context_text = paragraphs['context']
			answer_info = paragraphs['qas'][qas_idx]['answers'][0]
			answer_start = answer_info['answer_start']
			answer_text = answer_info['text']
			ground_truth = paragraphs['qas'][qas_idx]['question'].split(" ")
			qas_id = paragraphs['qas'][qas_idx]['id']
			question_context_answer.append((context_text, answer_text, answer_start, ground_truth, qas_id))
			overall_qas_idx += 1

root = tk.Tk()
my_gui = MainGUI(root, question_context_answer)
root.mainloop()

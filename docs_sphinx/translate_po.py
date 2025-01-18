import deepl
import polib
import dotenv
import os

dotenv.load_dotenv()

translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
po_file = 'source/locales/pl/LC_MESSAGES/modules.po'
po = polib.pofile(po_file)
strings_for_no_translation = ["list", "callable", ]

for entry in po:
    if not entry.translated():
        if entry.msgid == "list":
            continue
        translated = translator.translate_text(entry.msgid, target_lang="PL").text
        entry.msgstr = translated
        print(f"Tłumaczę: {entry.msgid} -> {translated}")

po.save(po_file)

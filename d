[1mNAME[0m
    synth_doc_generation.py

[1mSYNOPSIS[0m
    synth_doc_generation.py [4mGROUP[0m | [4mCOMMAND[0m | [4mVALUE[0m

[1mGROUPS[0m
    [1m[4mGROUP[0m[0m is one of the following:

     pathlib

     re
       Support for regular expressions (RE).

     json
       JSON (JavaScript Object Notation) <https://json.org> is a subset of JavaScript syntax (ECMA-262 3rd edition) used as a lightweight data interchange format.

     asyncio
       The asyncio package, tracking PEP 3156.

     logging
       Logging package for Python. Based on PEP 282 and comments thereto in comp.lang.python.

     random
       Random variable generators.

     fire
       The Python Fire module.

     time
       This module provides various functions to manipulate time values.

     os
       OS routines for NT or Posix depending on what system we're on.

     safetytooling_utils

     LOGGER
       Instances of the Logger class represent a single logging channel. A "logging channel" indicates an area of an application. Exactly how an "area" is defined is up to the application developer. Since an application can have any number of areas, logging channels are identified by a unique string. Application areas can be nested (e.g. an area of "input processing" might include sub-areas "read CSV files", "read XLS files" and "read Gnumeric files"). To cater for this natural nesting, channel names are organized into a namespace hierarchy where levels are separated by periods, much like the Java or Python package namespace. So in the instance given above, channel names might be "input" for the upper level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels. There is no arbitrary limit to the depth of nesting.

[1mCOMMANDS[0m
    [1m[4mCOMMAND[0m[0m is one of the following:

     tqdm
       Asynchronous-friendly version of tqdm.

     Progress
       Renders an auto-updating progress bar(s).

     InferenceAPI
       A wrapper around the OpenAI and Anthropic APIs that automatically manages rate limits and valid responses.

     ChatMessage
       !!! abstract "Usage Documentation" [Models](../concepts/models.md)

     MessageRole
       str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str

     Prompt
       !!! abstract "Usage Documentation" [Models](../concepts/models.md)

     BatchInferenceAPI
       A wrapper around batch APIs that can do a bunch of model calls at once.

     dedent
       Remove any common leading whitespace from every line in `text`.

     UniverseContext
       !!! abstract "Usage Documentation" [Models](../concepts/models.md)

     SynthDocument
       !!! abstract "Usage Documentation" [Models](../concepts/models.md)

     load_txt

     parse_tags
       Parse text between opening and closing tags with the given tag name.

     load_jsonl

     load_json

     pst_time_string

     SyntheticDocumentGenerator

     abatch_augment_synth_docs

     aaugment_synth_docs

     abatch_generate_documents

     agenerate_documents

     batch_generate_documents_from_doc_specs

[1mVALUES[0m
    [1m[4mVALUE[0m[0m is one of the following:

     HOME_DIR

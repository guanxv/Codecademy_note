#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#- VS code  VScode visual studio code. #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#short cut key

#command line ctrl + `

#setting ctrl + ,

#search anything
#ctrl + shift + p 

#code formating
#alt + shift + f

#multi line editing
#ctrl + alt + arrow key

#indentation 
#ctrl + [
#ctrl + ]

#run code
#ctrl + alt + n

#toggle comment block
# ctrl + /

#output 中文乱码 （在代码中加入）
#--------------解决 VScode output 里中文乱码的问题---------------------
import io
import sys
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
#--------------解决 VScode output 里中文乱码的问题---------------------








#------------ my vscode setup file, setting.json ----------------


{
    "files.maxMemoryForLargeFilesMB": 16384,
    "editor.codeLens": false,
    "workbench.iconTheme": "ayu",
    // Determines which settings editor to use by default.
    //  - ui: Use the settings UI editor.
    //  - json: Use the JSON file editor.
    "workbench.settings.editor": "json",
    "workbench.settings.openDefaultSettings": true,
    "workbench.startupEditor": "newUntitledFile",
    "python.formatting.provider": "black",
    "editor.formatOnSave": false,
    "code-runner.clearPreviousOutput": true,
    "code-runner.showExecutionMessage": false,
    "code-runner.executorMap": {
        "python": "$pythonPath -u $fullFileName"
    },
    "terminal.integrated.shell.windows": "C:\\Program Files\\Git\\bin\\bash.exe",
    "git.autofetch": true,

    // Automatically start Kite Engine on editor startup if it's not already running.
    "kite.startKiteEngineOnStartup": true,
    
    // Whether or not to show the Kite welcome notification on startup.
    "kite.showWelcomeNotificationOnStartup": false,
    
    // Whether to save the current file before running.
    "code-runner.saveFileBeforeRun": true,
    "sqltools.useNodeRuntime": true,

    // Controls whether suggestions should automatically show up while typing.
    "editor.quickSuggestions": {
		"other": false,
		"comments": false,
		"strings": false
	},
}

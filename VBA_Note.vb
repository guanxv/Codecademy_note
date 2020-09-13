VBA Webscraping Note

'VBA can only work with IE
'seletc the VBA Microsoft HTML library


'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' web scraping initilization

'URL for login
Login_URL = "https://secure.workforceready.com.au/ta/PL112625.login?rnd=UIO"

'Initiate the IE
Dim appIE As Object
Set appIE = CreateObject("internetexplorer.application")

With appIE
    .Navigate Login_URL
    .Visible = True
End With

'wair for IE to load the page + 2 Second
Do While appIE.Busy
    DoEvents
Loop

Application.Wait Now + #12:00:02 AM#

Dim HTMLDoc As htmldocument
Set HTMLDoc = appIE.document

' another way of wait for IE to Load 

Do while appIE.Busy = True or appIE.ReadyState <> READYSTATE_COMPLETE

	Application.Wait Now + TimeValue("00:00:02")

Loop


'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' print the current URL we are at

debug.print appIE.LocaitonURL

'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' some times the web page can have a inner page, the source code looks like bellow. 

'<frame name="HCM_CENTER" frameborder="0" border="0" scrolling="no" src="/ta/PL112625.hcm?rnd=UJZ&amp;@impl=zeyt.web.UiControl_Blank&amp;@windowId=PCATQ&amp;Ext=login&amp;sft=NFJTYFIGLQ&amp;@pushOnStack=false&amp;@isPreservedFrame=1">
'#document
'<html><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><meta name="viewport" content="user-scalable=no, initial-scale=1, maximum-scale=1, minimum-scale=1, width=device-width, viewport-fit=contain"><link async="" rel="stylesheet" type="text/css" href="/ta/css/17410998/77/13/-80/webapps.css?CompId=17410998&amp;RND=v70&amp;CompId=17410998">........................

'to access the inner html element , you need to do this

Dim hamburgMenu As htmldocument
Set hamburgMenu = HTMLDoc.querySelector("[name=HCM_CENTER]").contentDocument


'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

'Fill in login info
HTMLDoc.all.UserName.value = "16.Guan"
HTMLDoc.all.PasswordView.value = "*******"
HTMLDoc.all.LoginButton.Click

'UserName , PasswordView, LoginButton is the "name" of the element.



'<div class="tab-content" id="tab-favorites">
'div is a tagName

'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
'find any thing by tagName , Tagname canbe :
	'frame
	'style
	'a  (for anchor)
	'body
	'div
	'li
	'ul
	
	
timesheet_keyword = "TimeSheetContainer"

For Each anchor In hamburgMenu.getElementsByTagName("a")

    If InStr(anchor.href, timesheet_keyword) <> 0 Then anchor.Click: Exit For
    
Next

'InStr help to check if the herf link contain the keyword.

'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' Find element by Tagname sampel 2, 


Dim myInfoButton As HTMLButtonElement
Set myInfoButton = hamburgMenu.getElementsByTagName

For Each Button In hamburgMenu.getElementsByTagName("button")
    If Button.Title = "My Info" Then Button.Click: Exit For
Next

'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' Find element by Tagname sampel 3, 
Debug.Print objIE.document.getElementsByTagName("p")(0).innerHTML

   'displays inner HTML of 1st p element on a page.
   
Debug.Print objIE.document.getElementsByTagName("p").Length

   'displays number of p (paragraph) elements on a page in the console.

   
   
'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' Find element by id, 

Dim hamb_icon As IHTMLElement
Set hamb_icon = adminMenu.getElementById("hamburgNav-show")

hamb_icon.Click

'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' get group of element
Dim timesheetul As IHTMLElementCollection
Set timesheetul = hamburgMenu.getElementsByClassName("top-level")
debug.print timesheetul.length ' check how many was selected

Dim secondul as IHTMLElement
Set secondul =  timesheetul.item(,2) ' select the second one

	debug.print secondul.Title
	debug.print secondul.calssName
	debug.print secondul.innerHTML
	debug.print secondul.innerText
	


'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
'loop through certain type of element.

Dim anchor As HTMLAnchorElement
For Each anchor In hamburgMenu.all
	debug.print achor.herf
Next

'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
'grab all anchors.

Dim IEAnchors as IHTMLElementCollection
Set IEAnchors = IEDocument.anchors

'grab a specific anchor
Dim IEAnchor as IHTMLAnchorElement
Set IEAchor = IEAnchors.item(,1) 'or item(1)??

	debug.print IEAnchor.host
	debug.print IEAnchor.hostname
	debug.print IEAnchor.pathname
	debug.print IEAnchor.herf
	debug.print IEAnchor.protocol
	
'grab all images
Dim IEImages as IHTMLElementCollection
Dim IEImage as IHTMLImgElement

Set IEImages = IEDocument.images
Set IEImage = IEImages.Item(1)

	debug.print IEImage.scr
	debug.print IEImage.fileCreatedDate
	debug.print IEImage.Height


	
'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
'
'HTMLElementCollection VS IHTMLElementCollection ???

'IHTMLElementCollection works most of the time. 
'HTMLElementCollection doesnt

'whats the different ???

	
'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' when finish scrapting close the application 



Set appIE = Nothing


End Function



'------------------------------------------


Function ADP_Timesheet_Submit()

'URL for login
Login_URL = "https://secure.workforceready.com.au/ta/PL112625.login?rnd=UIO"

'Initiate the IE
Dim appIE As Object
Set appIE = CreateObject("internetexplorer.application")

With appIE
    .Navigate Login_URL
    .Visible = True
End With

'wair for IE to load the page
Do While appIE.Busy
    DoEvents
Loop


Dim HTMLDoc As htmldocument
Set HTMLDoc = appIE.document


'Fill in login info
HTMLDoc.all.UserName.value = "16.Guan"
HTMLDoc.all.PasswordView.value = "0p;/)P:?"
HTMLDoc.all.LoginButton.Click

'wair for IE to load the page
Do While appIE.Busy
    DoEvents
Loop

Application.Wait Now + #12:00:02 AM#


'get hambuger menu inner html
Dim hamburgMenu As htmldocument
Set hamburgMenu = HTMLDoc.querySelector("[name=HAMBURG_MENU]").contentDocument

'find the url for timesheet page
timesheet_keyword = "TimeSheetContainer"

For Each anchor In hamburgMenu.getElementsByTagName("a")

    If InStr(anchor.href, timesheet_keyword) <> 0 Then anchor.Click: Exit For 'pay attention to the exit for, once it found the first element, i will stop looping
    
Next

'wait for page to be loaded
Do While appIE.Busy
    DoEvents
Loop

Application.Wait Now + #12:00:02 AM#

'get the inner Html for the table

Dim timesheetSubFrame As htmldocument
Set timesheetSubFrame = HTMLDoc.querySelector("[name=HCM_CENTER]").contentDocument

'get the inner html for the table

Dim timesheetSubSubFrame As htmldocument
Set timesheetSubSubFrame = timesheetSubFrame.querySelector("[name=SPA_CENTER]").contentDocument

Dim timesheetTable As HTMLTableSection
Set timesheetTable = timesheetSubSubFrame.getElementsByClassName("c-table__body").Item(, 0)

'expand all the date
Dim expandButtons As IHTMLElementCollection
Set expandButtons = timesheetSubSubFrame.getElementsByTagName("button")

For Each Button In expandButtons
    If Button.Title = "Expand" Then Button.Click

Next

'add empty record line
Dim addRecButtons As IHTMLElementCollection
Set addRecButtons = timesheetSubSubFrame.getElementsByTagName("button")

For Each Button In addRecButtons
    Debug.Print Button.Name
    Debug.Print Button.className
    If Button.className = "btn-meta low small" Then Button.Click
    
Next




Set appIE = Nothing


End Function




'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' Excel VBA work with cells

'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
'loop through each cell in a range.

 Set rng = Worksheets("KBSTimesheet").Range("E8:AC14")
 
     For Each Row In rng.Rows
        
        For Each cell In Row.Cells
            
            If Not cell.Text = "" Then Count = Count + 1
    
        Next cell
            
    Next Row

	
'find certain cells



.Range("A1:A4")'multiple cells	, 	$A$1:$A$4 
.Cells(1,5)	' Cells	row, column	one cell	, $E$1 
Range("A1:A2").Offset(1,2) 'Offset	row, column	multiple cells	, $C$2:$C$3
.Rows(4)' , Rows	row(s)	one or more rows
.Rows("2:4")' $4:$4 $2:$4	
.Columns(4) ' Columns	column(s)	one or more columns	
.Columns("B:D")	' $D:$D, $B:$D
 



'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
' Excel VBA general
'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

'work with list / array

Dim numOfRecordPerDay(1 To 7) As Integer

Dim array_sample(365) as single

Dim varData(3) As Variant 

Dim sngMulti(1 To 5, 1 To 10) As Single 

numOfRecordPerDay(1) = 10
numOfRecordPerDay(2) = 20

varData(0) = "Claudia Bendel" 
varData(1) = "4242 Maple Blvd" 
varData(2) = 38 
varData(3) = Format("06-09-1952", "General Date") 

sngMulti(1,4) = 0.54321

for each num in numOfRecordPerDay

	debug.print num
	
	


'#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
'do while loop sample
	
	
   Do While i < 5
      i = i + 1
      msgbox "The value of i is : " & i
   Loop


'error handling
'sample 1
	      
    On Error Resume Next 'will keep running when error raised after this line. 
							'if error happend before this line , program will stop.
     
    a = 10 / 0

    
    If Err.Number = 0 Then  ' if error number = 0 means no error. 
							' if error number <>0, means there are some errors, you have to jump then.

        Debug.Print "ok"
        a = 0
    
    End If
	
	On error goto 0 'use this one to stop the error handler. 
	
	
'funciton return a array

public funtion func_return_array() as variant

	dim result as variant
	
	result(1) = 1
	result(2) = 2
	
	func_return_array = result
	
end function


dim var as variant
var = func_return_array








'----------------------porject ---------------------------

'automatic fill timesheet on workforceready web page.


Function ADP_Timesheet_Submit()

'URL for login
Login_URL = "https://secure.workforceready.com.au/ta/PL112625.login?rnd=UIO"

'Initiate the IE
Dim appIE As Object
Set appIE = CreateObject("internetexplorer.application")

With appIE
    .Navigate Login_URL
    .Visible = True
End With

'wair for IE to load the page
Do While appIE.Busy
    DoEvents
Loop


Dim HTMLDoc As htmldocument
Set HTMLDoc = appIE.document


'Fill in login info
HTMLDoc.all.UserName.value = "16.Guan"
HTMLDoc.all.PasswordView.value = "0p;/)P:?"
HTMLDoc.all.LoginButton.Click

'wair for IE to load the page
Do While appIE.Busy
    DoEvents
Loop

Application.Wait Now + #12:00:03 AM#


'get hambuger menu inner html
Dim hamburgMenu As htmldocument
Set hamburgMenu = HTMLDoc.querySelector("[name=HAMBURG_MENU]").contentDocument

'find the url for timesheet page
timesheet_keyword = "TimeSheetContainer"

For Each anchor In hamburgMenu.getElementsByTagName("a")

    If InStr(anchor.href, timesheet_keyword) <> 0 Then anchor.Click: Exit For 'pay attention to the exit for, once it found the first element, i will stop looping
    
Next

'wait for page to be loaded
Do While appIE.Busy
    DoEvents
Loop

Application.Wait Now + #12:00:03 AM#

Set HTMLDoc = appIE.document


'get the inner Html for the table

Dim timesheetSubFrame As htmldocument
Set timesheetSubFrame = HTMLDoc.querySelector("[name=HCM_CENTER]").contentDocument

'get the inner html for the table

Dim timesheetSubSubFrame As htmldocument
Set timesheetSubSubFrame = timesheetSubFrame.querySelector("[name=SPA_CENTER]").contentDocument

Dim timesheetTable As HTMLTableSection
Set timesheetTable = timesheetSubSubFrame.getElementsByClassName("c-table__body").Item(, 0)

'expand all the date
Dim Buttons As IHTMLElementCollection
Set Buttons = timesheetSubSubFrame.getElementsByTagName("button")

For Each Button In Buttons
    If Button.Title = "Expand" Then Button.Click  ' add expand the day
    'If Button.Title = "Collapse" Then Button.Click 'collapse the day
Next


'and delete all the existing line
For Each Button In Buttons
    If Button.getAttribute("aria-label") = "delete" Then Button.Click
Next


'add empty record line base on timesheet.
Dim addRecButtons As IHTMLElementCollection
Set addRecButtons = timesheetSubSubFrame.getElementsByTagName("button")

Call addentryline(addRecButtons, numRecordperday)

Dim rawTolnum As Variant
rawTolnum = getTimesheetRawTot

For Each num In rawTolnum
    Debug.Print num
Next

Dim count As Integer
count = 0

'find input block for raw data
Dim inputs As IHTMLElementCollection
Set inputs = timesheetSubSubFrame.getElementsByTagName("input")

    For Each ipt In inputs
        
        If ipt.getAttribute("placeholder") = "Raw Total" Then
        
            On Error Resume Next
            
                ipt.value = rawTolnum(Int(count / 2))
                
            If Err.Number = 0 Then count = count + 1
                        
            On Error GoTo 0
              
        End If
    Next
    
    
' find out the job names and type of jobs

Dim jobNames As Variant
jobNames = getTimesheetJobName

Dim jobnum(180) As String
Dim jobtype(180) As String

count = 0

For Each Name In jobNames

    If Len(Name) > 5 Then
    
        jobnum(count) = Left(Name, 6)
        jobtype(count) = Right(Name, Len(Name) - 6)
        
        count = count + 1
        
    Else
    
        jobnum(count) = ""
        jobtype(count) = Name
        count = count + 1
        
    End If
    
Next


count = 0

For Each typ In jobtype

    Select Case typ
    
        Case "M"
            If jobnum(count) = "" Then jobtype(count) = "Miscellaneou"
        
        Case "S"
            If jobnum(count) = "" Then jobtype(count) = "Sales (S)"
            
        Case "g"
            If jobnum(count) = "Meetin" Then
                
                jobtype(count) = "Meeting"
                jobnum(count) = ""
            
            End If
                
    End Select
    
    count = count + 1
    
Next

        
'fill in the job names and types

    countjob = 0
    counttype = 0
    total_count = 0

    For Each ipt In inputs
        
        If ipt.getAttribute("placeholder") = "Choose..." Then
            
            If total_count Mod 2 = 0 Then
            
                ipt.value = jobnum(countjob)
                total_count = total_count + 1
                countjob = countjob + 1
                
            Else
            
                ipt.value = jobtype(counttype)
                total_count = total_count + 1
                counttype = counttype + 1
                         
              
            End If
        End If
    Next

     


Set appIE = Nothing


End Function


Function numRecordperday() As Integer()

    Dim numOfRecordPerDay(1 To 7) As Integer
    Dim rng As Range
    
    Set rng = Worksheets("KBSTimesheet").Range("E8:AC14")
        
    daynum = 1
        
    For Each Row In rng.Rows
    
        count = 0
        
        
        For Each cell In Row.Cells
            
            If Not cell.Text = "" Then count = count + 1
    
        Next cell
        
        numOfRecordPerDay(daynum) = count
        
        daynum = daynum + 1
    
        
    Next Row
    
    
    'For Each Number In numOfRecordPerDay
    '    Debug.Print Number
    'Next
    
    numRecordperday = numOfRecordPerDay

End Function



Function addentryline(Buttons As IHTMLElementCollection, num() As Integer)

    daynum = 6
    
    For Each Button In Buttons
        
        If daynum = 8 Then daynum = 1
          
        'If Button.className = "btn-meta low small" And Button.getAttribute("aria-label") = "Add Time Entry" Then
        If Button.className = "btn-meta low small" Then
            
            count = num(daynum)
            
            Do While count <> 0
            
                
                'Debug.Print "daynum = "
                'Debug.Print daynum
                'Debug.Print "Total = "
                'Debug.Print num(daynum)
                'Debug.Print " Click "
                'Debug.Print " Count = "
                'Debug.Print Count
                
                Button.Click
                
                count = count - 1
                
            Loop
            
        daynum = daynum + 1
        
        End If
        
        
        
    Next
    
    
End Function



Function getTimesheetRawTot() As Variant

    Dim rawtot(180) As Variant
    Dim rng As Range
    
    Set rng1 = Worksheets("KBSTimesheet").Range("E13:AC14")
    Set rng2 = Worksheets("KBSTimesheet").Range("E8:AC12")
            
    Dim coun As Integer
    count = 0
    
    'check saturday sunday first to match the ADP webpage
            
    For Each Row In rng1.Rows
        
        For Each cell In Row.Cells
            
            If Not cell.Text = "" Then
                
                rawtot(count) = cell.value * 24
                
                count = count + 1
            
            End If
    
        Next cell
        
    Next Row
    
   'check weekdays
    
    For Each Row In rng2.Rows
        
        For Each cell In Row.Cells
            
            If Not cell.Text = "" Then
                
                rawtot(count) = cell.value * 24
                
                count = count + 1
            
            End If
    
        Next cell
        
    Next Row
    
    
    
    'For Each num In rawtot
    '    Debug.Print num
    'Next
    
 
    getTimesheetRawTot = rawtot
    
    
    
End Function



Function getTimesheetJobName() As Variant

    Dim jobName(180) As Variant
    Dim rng As Range
    
    Set rng1 = Worksheets("KBSTimesheet").Range("E13:AC14")
    Set rng2 = Worksheets("KBSTimesheet").Range("E8:AC12")
            
    Dim coun As Integer
    count = 0
    
    'check saturday sunday first to match the ADP webpage
            
    For Each Row In rng1.Rows
        
        For Each cell In Row.Cells
            
            If Not cell.Text = "" Then
                
                jobName(count) = Worksheets("KBSTimesheet").Cells(7, cell.Column)
                
                count = count + 1
            
            End If
    
        Next cell
        
    Next Row
    
   'check weekdays
    
    For Each Row In rng2.Rows
        
        For Each cell In Row.Cells
            
            If Not cell.Text = "" Then
                
                jobName(count) = Worksheets("KBSTimesheet").Cells(7, cell.Column)
                
                count = count + 1
            
            End If
    
        Next cell
        
    Next Row
    
    
    
    'For Each Name In jobName
        'Debug.Print Name
    'Next
    
 
    getTimesheetJobName = jobName
    
    
    
End Function










	
    






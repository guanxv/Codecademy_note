#!/bin/bash


###### update on 2022.01.22 ############
#google provide a service to exprot all the setting and information
#see https://takeout.google.com/settings/takeout for more detail


############ to run this command Copy this command in terminal ########
#~/Desktop/google_photo_backup_to_pc.sh
#or./google_photo_backup_to_pc.sh (when standing at the desktop)

export PATH="$PATH:~/.local/bin"
cd ~/Python_Project/gphotos-sync/






##### SOLVE FOR MOUNTING HARD DRIVE ISSUE ######

#sudo blkid
#find the drive to mount like this one /dev/sda1: UUID="5FE9860853F8E5F8" TYPE="ntfs" PTTYPE="atari" PARTLABEL="3T Storage" PARTUUID="cd9dd8da-5136-4fb1-9ab2-a555e8ff0d5a"
#sudo mkdir /media/user/5FE9860853F8E5F8
#sudo nano /etc/fstab
#add this at bottom , UUID=5FE9860853F8E5F8 /media/user/5FE9860853F8E5F8 ntfs defaults 0 0
#save, exit, restart

#see ref https://askubuntu.com/questions/365052/how-to-mount-drive-in-media-username-like-nautilus-does-using-udisks

##### END #######





##### SOLVE FOR BAD LINK / ISSUE, #######

# A. DELETE THE gphotos.bad_ids.yaml FILE UNDER THE ROOT FOLDER.

# B. RUN THE SYNC WITH --rescan 

#pipenv run gphotos-sync --rescan --progress /media/user/5FE9860853F8E5F8/Home/gphotos-sync

##### END #######




##### Normal Daily Sync command ##############

pipenv run gphotos-sync --progress /media/user/5FE9860853F8E5F8/Home/gphotos-sync

##### END #######








##### (invalid_grant) Bad Request ##############

#if you got an error of  // (invalid_grant) Bad Request , use below line for a new token

#pipenv run gphotos-sync --new-token /media/user/5FE9860853F8E5F8/Home/gphotos-sync

##### END #######





##### (invalid_client) Unauthorized ##############

#if you got an error of oauthlib.oauth2.rfc6749.errors.InvalidClientError: (invalid_client) Unauthorized

#you might need to update the client_secret.json
#goto https://console.cloud.google.com/apis/credentials/
#find gphoto-client
#reset key
#downlaod the json file
#rename the file to client_secret.json and drop it here /home/user/.config/gphotos-sync
#run this command
#pipenv run gphotos-sync --new-token /media/user/5FE9860853F8E5F8/Home/gphotos-sync
#click the link and copy the token
#paste token back

##### END #######




#rescan entire library, ignoring last scan date. Use this if you have added photos to the library that predate the last sync, or you have deleted some of the local files
#pipenv run gphotos-sync /media/user/5FE9860853F8E5F8/Home/gphotos-sync --rescan 


#check for the existence of files marked as already downloaded and re-download any missing ones. Use this if you have deleted some local files

# pipenv run gphotos-sync --archived --retry-download --skip-index --progress /media/user/5FE9860853F8E5F8/Home/gphotos-sync




#delete the index db, re-scan everything

#pipenv run gphotos-sync /media/user/5FE9860853F8E5F8/Home/gphotos-sync --flush-index 


### to run this command Copy this command in terminal
#~/Desktop/google_photo_backup_to_pc.sh
#or./google_photo_backup_to_pc.sh (when standing at the desktop)


# '''
# usage: gphotos-sync [-h] [--album ALBUM | --album-regex REGEX] [--log-level LOG_LEVEL] [--logfile LOGFILE]
#                     [--compare-folder COMPARE_FOLDER] [--favourites-only] [--flush-index] [--rescan]
#                     [--retry-download] [--skip-video] [--skip-shared-albums] [--album-date-by-first-photo]
#                     [--start-date START_DATE] [--end-date END_DATE] [--db-path DB_PATH] [--albums-path ALBUMS_PATH]
#                     [--photos-path PHOTOS_PATH] [--use-flat-path] [--omit-album-date] [--new-token] [--index-only]
#                     [--skip-index] [--do-delete] [--skip-files] [--skip-albums] [--use-hardlinks] [--no-album-index]
#                     [--case-insensitive-fs] [--max-retries MAX_RETRIES] [--max-threads MAX_THREADS] [--secret SECRET]
#                     [--archived] [--progress] [--max-filename MAX_FILENAME] [--ntfs]
#                     root_folder

# Google Photos download tool

# positional arguments:
#   root_folder           root of the local folders to download into

# optional arguments:
#   -h, --help            show this help message and exit
#   --album ALBUM         only synchronize the contents of a single album. use quotes e.g. "album name" for album names
#                         with spaces
#   --album-regex REGEX   only synchronize albums that match regular expression. regex is case insensitive and
#                         unanchored. e.g. to select two albums: "^(a full album name|another full name)$"
#   --log-level LOG_LEVEL
#                         Set log level. Options: critical, error, warning, info, debug, trace. trace logs all Google
#                         API calls to a file with suffix .trace
#   --logfile LOGFILE     full path to debug level logfile, default: <root>/gphotos.log. If a directory is specified
#                         then a unique filename will be generated.
#   --compare-folder COMPARE_FOLDER
#                         root of the local folders to compare to the Photos Library
#   --favourites-only     only download media marked as favourite (star)
#   --flush-index         delete the index db, re-scan everything
#   --rescan              rescan entire library, ignoring last scan date. Use this if you have added photos to the
#                         library that predate the last sync, or you have deleted some of the local files
#   --retry-download      check for the existence of files marked as already downloaded and re-download any missing
#                         ones. Use this if you have deleted some local files
#   --skip-video          skip video types in sync
#   --skip-shared-albums  skip albums that only appear in 'Sharing'
#   --album-date-by-first-photo
#                         Make the album date the same as its earliest photo. The default is its last photo
#   --start-date START_DATE
#                         Set the earliest date of files to syncformat YYYY-MM-DD
#   --end-date END_DATE   Set the latest date of files to syncformat YYYY-MM-DD
#   --db-path DB_PATH     Specify a pre-existing folder for the index database. Defaults to the root of the local
#                         download folders
#   --albums-path ALBUMS_PATH
#                         Specify a folder for the albums Defaults to the 'albums' in the local download folders
#   --photos-path PHOTOS_PATH
#                         Specify a folder for the photo files. Defaults to the 'photos' in the local download folders
#   --use-flat-path       Mandate use of a flat directory structure ('YYYY-MMM') and not a nested one ('YYYY/MM') .
#   --omit-album-date     Don't include year and month in album folder names.
#   --new-token           Request new token
#   --index-only          Only build the index of files in .gphotos.db - no downloads
#   --skip-index          Use index from previous run and start download immediately
#   --do-delete           Remove local copies of files that were deleted. Must be used with --flush-index since the
#                         deleted items must be removed from the index
#   --skip-files          Dont download files, just refresh the album links (for testing)
#   --skip-albums         Dont download albums (for testing)
#   --use-hardlinks       Use hardlinks instead of symbolic links in albums and comparison folders
#   --no-album-index      only index the photos library - skip indexing of folder contents (for testing)
#   --case-insensitive-fs
#                         add this flag if your filesystem is case insensitive
#   --max-retries MAX_RETRIES
#                         Set the number of retries on network timeout / failures
#   --max-threads MAX_THREADS
#                         Set the number of concurrent threads to use for parallel download of media - reduce this
#                         number if network load is excessive
#   --secret SECRET       Path to client secret file (by default this is in the application config directory)
#   --archived            Download media items that have been marked as archived
#   --progress            show progress of indexing and downloading in warning log
#   --max-filename MAX_FILENAME
#                         Set the maxiumum filename length for target filesystem.This overrides the automatic detection.
#   --ntfs                Declare that the target filesystem is ntfs (or ntfs like).This overrides the automatic
#                         detection.
#                         '''


#!/bin/bash

APP=symbol-classification
PID=/tmp/$APP.pid
LOG=/tmp/$APP.log
ERROR=/tmp/$APP-error.log

PORT=8888
VENV="source /home/muret/venv/bin/activate"
COMMAND="python /home/muret/symbol-classification/server.py -port $PORT"

#USR=muret

status() {
    echo
    echo "==== Status"

    if [ -f $PID ]
    then
        echo
        echo "Pid file: $( cat $PID ) [$PID]"
        echo
        ps -ef | grep -v grep | grep $( cat $PID )
    else
        echo
        echo "No Pid file"
    fi
}

start() {
    if [ -f $PID ]
    then
        echo
        echo "Already started. PID: [$( cat $PID )]"
    else
        echo "==== Start"
        touch $PID
        $VENV
        if nohup $COMMAND >>$LOG 2>&1 &
        then echo $! >$PID
             echo "Done."
             echo "$(date '+%Y-%m-%d %X'): START" >>$LOG
        else echo "Error... "
             /bin/rm $PID
        fi
    fi
}

kill_cmd() {
    SIGNAL=""; MSG="Killing "
    while true
    do
        #LIST=`ps -ef | grep -v grep | grep $CMD | grep -w $USR | awk '{print $2}'`
        #if [ "$LIST" ]
        if [ -f $PID ]
        then
            echo; echo "$MSG $LIST" ; echo
            echo $LIST | xargs kill $SIGNAL
            sleep 2
            SIGNAL="-9" ; MSG="Killing $SIGNAL"
            if [ -f $PID ]
            then
                /bin/rm $PID
            fi
        else
           echo; echo "All killed..." ; echo
           break
        fi
    done
}

stop() {
    echo "==== Stop"

    if [ -f $PID ]
    then
        if kill $( cat $PID )
        then echo "Done."
             echo "$(date '+%Y-%m-%d %X'): STOP" >>$LOG
        fi
        /bin/rm $PID
        kill_cmd
    else
        echo "No pid file. Already stopped?"
    fi
}

case "$1" in
    'start')
            start
            ;;
    'stop')
            stop
            ;;
    'restart')
            stop ; echo "Sleeping..."; sleep 1 ;
            start
            ;;
    'status')
            status
            ;;
    *)
            echo
            echo "Usage: $0 { start | stop | restart | status }"
            echo
            exit 1
            ;;
esac

exit 0

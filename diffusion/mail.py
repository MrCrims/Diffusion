import smtplib
from email.mime.text import MIMEText
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide message to send."
    )
    parser.add_argument(
        '-t', '--task', 
        type=str, 
        default='train',
        help="The name of submitted task",
    )
    return parser.parse_args()


if __name__ == '__main__':
    '''
    command: python your_mail_script.py -t your_task_name
    '''

    args = parse_args()
    # print(args.task)
    msg_from="mrcrimson@163.com"  #填入发送方邮箱
    pwd="QCRCKWEGKDLTAGBB"       #填入发送方邮箱smtp授权码
    to="983056925@qq.com"        #填入发送方邮箱
    subject="CLUSTER-INFO"    #电子邮件的主题
    content=f"task {args.task} completed"        #电子邮件的内容

    #构造邮件（注意大写字母）
    msg=MIMEText(content)
    msg["Subject"]=subject
    msg["From"]=msg_from
    msg["To"]=to

    #发送邮件
    try:
        ss=smtplib.SMTP_SSL("smtp.163.com",465)   #发送方邮箱smtp安全协议
        ss.login(msg_from,pwd)                   #登录                 
        ss.sendmail(msg_from,to,msg.as_string()) #发送
        print("email sent successfully")
    except Exception as e:
        print("sending failed: ",e) 

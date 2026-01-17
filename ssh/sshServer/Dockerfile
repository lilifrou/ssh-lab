FROM lscr.io/linuxserver/openssh-server:latest

# Copy public key into expected location
COPY id_rsa.pub /.ssh/id_rsa.pub


#COPY start.sh /start.sh
#RUN chmod +x /start.sh
#CMD ["/start.sh"]


# Allow TCP Forwarding
RUN sed -i 's/^[#[:space:]]*AllowTcpForwarding.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config

# Optional: ensure directory exists and has correct perms (depends on base image)
RUN mkdir -p /.ssh && chmod 700 /.ssh && chmod 644 /.ssh/id_rsa.pub

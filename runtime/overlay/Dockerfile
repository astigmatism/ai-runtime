# This house's layer on the image built from Strata's own Dockerfile at the pinned revision.
#
# Two changes, both about running here rather than about what the engine serves:
#
# * /opt/strata is made world-writable, because this house runs every inference backend as the
#   deployment user and Strata's entry point writes its own config link and per-start log into the
#   install tree. It is the only change to the image that is built without vision.
# * the entry point is replaced by strata-vision-entrypoint.sh, which applies the encoder placement
#   this house declares and then execs the upstream entry point unchanged. Only the image that has
#   the vision encoder built in carries it, so the text-only image stays exactly what its receipt
#   says it is.
#
# The build receipt in ops/reports/<date>-strata-vision/ names the revision, both digests, and the
# sha256 of the entry point that went in.
ARG UPSTREAM
FROM ${UPSTREAM}
RUN chmod a+w /opt/strata
RUN mv /opt/strata/docker-entrypoint.sh /opt/strata/docker-entrypoint.upstream.sh
COPY strata-vision-entrypoint.sh /opt/strata/docker-entrypoint.sh
RUN chmod +x /opt/strata/docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]

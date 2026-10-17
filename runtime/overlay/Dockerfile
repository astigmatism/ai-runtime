# This house's layer on the image built from Strata's own Dockerfile at the pinned revision.
#
# Three changes, all about running here rather than about what the engine serves:
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
#
# The identity labels are declared here, and are not inherited: Strata's own Dockerfile labels only
# the NVIDIA base, so the image this house ships is the only place they exist, and building this
# layer without them loses them. That is not cosmetic. Four paths in the runtime read
# `org.opencontainers.image.revision` back out of `docker image inspect` -- the switch guard in
# controller.validate, the update cutover in update.py, the migration check in migration.py, and the
# recovery path in recovery.py -- and the first of them refuses a deploy whose image does not carry
# the revision the manifest pins, with the message `pinned inference engine identity changed`, after
# the candidate has built and the checkout has been read. None of the three has a default, and the
# RUN below stops a build that omits one, so the mistake surfaces as a failed build rather than as an
# image that runs perfectly and is refused at the gate -- which is how it was found the first time.
#
#   docker build -t local/strata:296b59d2f73c-vision \
#     --build-arg UPSTREAM=strata-upstream:296b59d2f73c-vision \
#     --build-arg REVISION=296b59d2f73c8f04a805df29c2fa97655adace78 \
#     --build-arg REPOSITORY=https://github.com/Niko1221/Strata \
#     --build-arg TRIAL=strata .
#
# Those are the values for the revision the engine entry pins today, and
# test_the_overlay_layer_declares_the_identity_the_runtime_reads_back fails if the revision or the
# repository here stops agreeing with that entry. TRIAL is the label Service Portal marks the images
# of a trial with; it is copied from the image in service rather than reasoned about, because nothing
# here is known to write it and something else may read it.
ARG UPSTREAM
ARG REVISION
ARG REPOSITORY
ARG TRIAL
FROM ${UPSTREAM}
# Repeated on purpose: an ARG declared before FROM is usable only by the FROM line, and BuildKit
# quietly expands an ARG the current stage has never declared to the empty string. Labels that
# expand to nothing build, ship and read back empty, which is the failure this paragraph is about --
# so all three are declared again for this stage, and the RUN below refuses them empty.
ARG REVISION
ARG REPOSITORY
ARG TRIAL
RUN test -n "$REVISION" -a -n "$REPOSITORY" -a -n "$TRIAL" || { echo "overlay build needs --build-arg REVISION=<the 40-character revision the manifest pins>, REPOSITORY=<its repository> and TRIAL=<the trial label the image in service carries>; an image built without them is refused at deploy" >&2; exit 1; }
LABEL org.opencontainers.image.revision=${REVISION}
LABEL org.opencontainers.image.source=${REPOSITORY}
LABEL io.service-portal.trial=${TRIAL}
RUN chmod a+w /opt/strata
RUN mv /opt/strata/docker-entrypoint.sh /opt/strata/docker-entrypoint.upstream.sh
COPY strata-vision-entrypoint.sh /opt/strata/docker-entrypoint.sh
RUN chmod +x /opt/strata/docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
